import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from config import Config
from core.enums import SignalType
import time
from binance.client import Client

# Profile Definitions
PROFILES = [
    {
        "name": "001_Guardian",
        "leverage": 1,
        "rsi_low": 30,
        "rsi_high": 70,
        "ema_filter": 200,
        "risk_pct": 0.005
    },
    {
        "name": "002_Maestro",
        "leverage": 3,
        "rsi_low": 35,
        "rsi_high": 65,
        "ema_filter": 200,
        "risk_pct": 0.01
    },
    {
        "name": "003_Hunter",
        "leverage": 5,
        "rsi_low": 40,
        "rsi_high": 60,
        "ema_filter": 100,
        "risk_pct": 0.02
    }
]

SYMBOLS = Config.CRYPTO_FUTURES_PAIRS[:20] # Top 20 gems
INITIAL_CAPITAL = 13.0
DAYS = 15

def get_data(client, symbol, days):
    """Fetch historical data from Binance."""
    limit = 1440 * days
    try:
        klines = client.get_historical_klines(
            symbol.replace('/', ''), 
            Client.KLINE_INTERVAL_1MINUTE, 
            f"{days} days ago UTC"
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def run_simulation(df, profile):
    """Run a simplified backtest for a specific profile."""
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = 0
    wins = 0
    max_drawdown = 0
    peak = capital
    
    # Pre-calculate indicator proxies
    closes = df['close'].values
    rsi_period = 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    ema_val = df['close'].rolling(window=profile['ema_filter']).mean().values
    
    in_position = False
    entry_price = 0
    direction = None
    
    for i in range(profile['ema_filter'], len(df)):
        price = closes[i]
        curr_rsi = rsi[i]
        curr_ema = ema_val[i]
        
        if not in_position:
            # Entry Logic
            # LONG: Price > EMA AND RSI < RSI_LOW
            if price > curr_ema and curr_rsi < profile['rsi_low']:
                in_position = True
                entry_price = price
                direction = "LONG"
                trades += 1
            # SHORT: Price < EMA AND RSI > RSI_HIGH
            elif price < curr_ema and curr_rsi > profile['rsi_high']:
                in_position = True
                entry_price = price
                direction = "SHORT"
                trades += 1
        else:
            # Exit Logic (Simple TP/SL or RSI reversal)
            pnl_pct = (price - entry_price) / entry_price if direction == "LONG" else (entry_price - price) / entry_price
            pnl_pct *= profile['leverage']
            
            # Simulated Exit Conditions
            exit_triggered = False
            if pnl_pct >= 0.015: # 1.5% TP
                exit_triggered = True
                wins += 1
            elif pnl_pct <= -0.02: # 2.0% SL
                exit_triggered = True
            elif (direction == "LONG" and curr_rsi > 70) or (direction == "SHORT" and curr_rsi < 30):
                exit_triggered = True
                if pnl_pct > 0: wins += 1
            
            if exit_triggered:
                # Update Capital
                trade_pnl = capital * profile['risk_pct'] * pnl_pct / 0.02 # Normalized risk
                capital += trade_pnl
                in_position = False
                
                # Metrics
                if capital > peak: peak = capital
                dd = (peak - capital) / peak
                if dd > max_drawdown: max_drawdown = dd
        
        equity_curve.append(capital)
    
    win_rate = (wins / trades * 100) if trades > 0 else 0
    net_pnl = capital - INITIAL_CAPITAL
    return {
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "max_dd": max_drawdown * 100,
        "total_trades": trades
    }

def main():
    client = Client()
    results = {}
    
    print(f"🚀 Starting Optimization Tournament (20 symbols, {DAYS} days)")
    
    all_data = {}
    for symbol in SYMBOLS:
        print(f"📥 Fetching data for {symbol}...")
        df = get_data(client, symbol, DAYS)
        if df is not None:
            all_data[symbol] = df
        time.sleep(1) # Slow down for API limits
    
    for profile in PROFILES:
        print(f"\n🏆 Testing Profile: {profile['name']}")
        profile_results = []
        for symbol, df in all_data.items():
            res = run_simulation(df, profile)
            profile_results.append(res)
        
        # Aggregate
        agg_pnl = sum(r['net_pnl'] for r in profile_results)
        agg_win = np.mean([r['win_rate'] for r in profile_results])
        agg_dd = max(r['max_dd'] for r in profile_results)
        agg_trades = sum(r['total_trades'] for r in profile_results)
        
        results[profile['name']] = {
            "Total PnL": agg_pnl,
            "Avg WinRate": agg_win,
            "Max DD": agg_dd,
            "Total Trades": agg_trades
        }
        
    print("\n" + "="*50)
    print("🏁 TOURNAMENT FINAL RESULTS")
    print("="*50)
    df_results = pd.DataFrame(results).T
    print(df_results)
    print("="*50)
    
    # Save to JSON
    with open("tests/tournament_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
