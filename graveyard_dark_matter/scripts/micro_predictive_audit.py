import os
import sys
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timezone, timedelta
import sqlite3
import argparse

# Configurar el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_forward_returns(exchange, symbol, entry_timestamp_ms, entry_price, direction):
    """
    Fetch 1m klines starting exactly at entry_timestamp to calculate
    Max Favorable Excursion (MFE) and Forward Returns at 1m, 3m, 5m, 15m.
    """
    try:
        # Fetch 20 1m candles starting exactly from the entry minute
        limit = 20
        since = int(entry_timestamp_ms) - 60000 # buffer of 1 min
        
        symbol_ccxt = symbol.replace('_', '') if not '/' in symbol else symbol
        
        ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe='1m', since=since, limit=limit)
        if not ohlcv or len(ohlcv) < 15:
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate returns based on direction
        mult = 1.0 if direction.upper() in ['LONG', 'BUY'] else -1.0
        
        returns = {}
        
        # Max Favorable Excursion in the next 15 minutes
        if mult == 1.0:
            mfe_price = df['high'].iloc[1:16].max()
        else:
            mfe_price = df['low'].iloc[1:16].min()
            
        returns['mfe_pct'] = ((mfe_price - entry_price) / entry_price * 100) * mult
        
        # Point-in-time returns
        try:
            returns['1m_pct'] = ((df['close'].iloc[1] - entry_price) / entry_price * 100) * mult
            returns['3m_pct'] = ((df['close'].iloc[3] - entry_price) / entry_price * 100) * mult
            returns['5m_pct'] = ((df['close'].iloc[5] - entry_price) / entry_price * 100) * mult
            returns['15m_pct'] = ((df['close'].iloc[15] - entry_price) / entry_price * 100) * mult
        except IndexError:
            # Not enough data yet (trade was too recent)
            return None
            
        return returns
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def analyze_predictive_decay():
    print("🔍 INICIANDO AUDITORÍA FORENSE PREDICTIVA (PREDICTION DECAY)")
    print("==============================================================")
    
    # 1. Load Trades
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data.db')
    
    trades_df = None
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            # Fetch closed trades or entry events. The table is usually 'trades' or 'positions'.
            # Let's see if we can get entry data. 'trades' table might have 'entry_time', 'symbol', 'direction', 'entry_price'
            query = "SELECT symbol, side as direction, price, timestamp as datetime, strategy_id FROM trades ORDER BY timestamp DESC LIMIT 50"
            trades_df = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"Error reading DB: {e}")
            
    if trades_df is None or trades_df.empty:
        print("No se encontraron trades en SQLite data.db. Intentando con el log de engine o fallback...")
        return
        
    print(f"✅ Cargados {len(trades_df)} trades de entrada para análisis.")
    
    # Sort by datetime
    trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
    trades_df = trades_df.sort_values('datetime').tail(50) # Analyze last 50 trades
    
    # 2. Connect to Binance
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    results = []
    
    print("\n⏳ Descargando datos de mercado tick-by-tick para cada trade...")
    for idx, row in trades_df.iterrows():
        symbol = row['symbol']
        direction = row['direction']
        entry_price = float(row['price'])
        strategy = row['strategy_id']
        
        # Convert timestamp to ms
        ts_ms = int(row['datetime'].timestamp() * 1000)
        
        print(f"   Analizando {symbol} | {direction} | Strat: {strategy}")
        
        fw_returns = fetch_forward_returns(exchange, symbol, ts_ms, entry_price, direction)
        
        if fw_returns:
            fw_returns['symbol'] = symbol
            fw_returns['direction'] = direction
            fw_returns['strategy'] = strategy
            results.append(fw_returns)
            
        time.sleep(0.5) # Rate limit respect
        
    if not results:
        print("❌ No se pudieron calcular retornos futuros (quizás los trades son muy recientes).")
        return
        
    res_df = pd.DataFrame(results)
    
    print("\n📊 RESULTADOS: EXACTITUD DIRECCIONAL Y DECAIMIENTO (PREDICTION DECAY)")
    print("=======================================================================")
    
    total = len(res_df)
    print(f"Trades analizados: {total}")
    
    # Exactitud = Win Rate a ese horizonte
    acc_1m = (res_df['1m_pct'] > 0).mean() * 100
    acc_3m = (res_df['3m_pct'] > 0).mean() * 100
    acc_5m = (res_df['5m_pct'] > 0).mean() * 100
    acc_15m = (res_df['15m_pct'] > 0).mean() * 100
    
    mfe_acc = (res_df['mfe_pct'] > 0.3).mean() * 100 # % de trades que superaron el +0.3% (Turbo BE)
    
    print(f"\n🎯 EXACTITUD PREDICTIVA (% de veces que el precio se mueve a nuestro favor):")
    print(f"  T+1  Minuto: {acc_1m:.1f}%")
    print(f"  T+3  Minutos: {acc_3m:.1f}%")
    print(f"  T+5  Minutos: {acc_5m:.1f}%")
    print(f"  T+15 Minutos: {acc_15m:.1f}%")
    
    print(f"\n🛡️ EXACTITUD MFE (Trades que dieron oportunidad de asegurar Breakeven al +0.3%):")
    print(f"  Win Rate MFE > 0.3%: {mfe_acc:.1f}%")
    
    print("\n📈 RETORNO PROMEDIO POR HORIZONTE:")
    print(f"  T+1m : {res_df['1m_pct'].mean():.3f}%")
    print(f"  T+3m : {res_df['3m_pct'].mean():.3f}%")
    print(f"  T+5m : {res_df['5m_pct'].mean():.3f}%")
    print(f"  T+15m: {res_df['15m_pct'].mean():.3f}%")
    print(f"  MFE Promedio: {res_df['mfe_pct'].mean():.3f}%")
    
    print("\n🧠 DESGLOSE POR ESTRATEGIA (MFE > 0.3%):")
    for strat, group in res_df.groupby('strategy'):
        strat_mfe_acc = (group['mfe_pct'] > 0.3).mean() * 100
        print(f"  - {strat} ({len(group)} trades): {strat_mfe_acc:.1f}% hit rate")

if __name__ == "__main__":
    analyze_predictive_decay()
