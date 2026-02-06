"""
📊 BACKTEST COMPLETO - Trader Gemini
Ejecuta backtest de 1 mes con datos reales de Binance

QUÉ: Script para validar la estrategia HybridScalpingStrategy
POR QUÉ: Verificar métricas antes de producción
PARA QUÉ: Confirmar Sharpe > 2.0, Drawdown < 1.5%
CÓMO: Descarga datos Binance → Simula trades → Calcula métricas
CUÁNDO: Antes de cada deployment a producción
DÓNDE: Se ejecuta localmente con datos históricos
QUIÉN: Risk Manager / QA Engineer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from queue import Queue
from binance.client import Client
from config import Config
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent, SignalEvent, OrderEvent
from core.enums import SignalType, OrderSide
from core.enums import SignalType, OrderSide
from data.data_provider import DataProvider
from core.market_regime import MarketRegimeDetector
from utils.logger import logger
import time

# ============================================================
# CONSTANTES
# ============================================================
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'] # Conservative Leaders
# FETCH REAL BALANCE
try:
    from binance.client import Client as RealClient
    try:
        real_client = RealClient(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY)
        acc = real_client.futures_account()
        real_bal = float(acc.get('totalWalletBalance', 0))
        INITIAL_CAPITAL = real_bal if real_bal > 0 else 15.0
        print(f"💰 REAL BALANCE DETECTED: ${INITIAL_CAPITAL:.2f}")
    except:
        INITIAL_CAPITAL = 15.0
        print(f"⚠️ Could not fetch balance, using default: ${INITIAL_CAPITAL}")
except:
    INITIAL_CAPITAL = 15.0

LEVERAGE = 5 # Conservative Leverage
COMMISSION_PCT = 0.0004 # 0.04% (Taker conservative)
RISK_PER_TRADE = 0.01  # 1% risk (Conservative)
DAYS = 15 # 15 Days validation (Longer horizon)

# ============================================================
# ESTRATEGIA SIMPLIFICADA PARA BACKTEST
# ============================================================

def calculate_simple_signal(bars: list, min_bars: int = 50) -> tuple:
    """
    Estrategia simplificada basada en BB + RSI + EMA
    Retorna: (signal_type, strength) o (None, 0)
    """
    if len(bars) < min_bars:
        return None, 0
    
    # Convertir a arrays
    closes = np.array([b['close'] for b in bars[-min_bars:]])
    highs = np.array([b['high'] for b in bars[-min_bars:]])
    lows = np.array([b['low'] for b in bars[-min_bars:]])
    
    current_price = closes[-1]
    
    # Bollinger Bands (20, 2)
    sma = np.mean(closes[-20:])
    std = np.std(closes[-20:])
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    bb_pct = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    # RSI (14)
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))
    
    # EMA cruce (9/21)
    ema9 = np.mean(closes[-9:])  # Aproximación simple
    ema21 = np.mean(closes[-21:])
    ema_bullish = ema9 > ema21
    
    # ATR para volatilidad
    # Necesitamos current High/Low (últimos 14) y previous Close (anteriores 14)
    current_highs = highs[-14:]
    current_lows = lows[-14:]
    prev_closes = closes[-15:-1]
    
    tr1 = current_highs - current_lows
    tr2 = np.abs(current_highs - prev_closes)
    tr3 = np.abs(current_lows - prev_closes)
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.mean(tr) if len(tr) > 0 else current_price * 0.01
    vol_pct = atr / current_price
    
    # Señales
    signal_type = None
    strength = 0.0
    
    # CONSERVATIVE / SMART LOGIC
    # 1. Trend Filter (EMA 200)
    ema200 = np.mean(closes[-200:]) if len(closes) >= 200 else closes[0]
    trend_bullish = current_price > ema200
    
    # 2. Stricter RSI (Only Extremes)
    # Buy dips in Uptrend, Sell rallies in Downtrend
    
    # LONG conditions
    # Rule: Price > EMA200 AND RSI < 30 (Pullback in Uptrend)
    if trend_bullish and rsi < 30:
        signal_type = SignalType.LONG
        strength = 0.8 + (30 - rsi) / 100
    elif rsi < 20: # Crash protection buy (Mean reversion rebound)
        signal_type = SignalType.LONG
        strength = 0.9
    
    # SHORT conditions
    # Rule: Price < EMA200 AND RSI > 70 (Rally in Downtrend) -- Less common in crypto bull runs but safer
    elif not trend_bullish and rsi > 70:
        signal_type = SignalType.SHORT
        strength = 0.8 + (rsi - 70) / 100
    elif rsi > 85: # Blow-off top sell
        signal_type = SignalType.SHORT
        strength = 0.9
        
    return signal_type, min(strength, 1.0)


# ============================================================
# CLASES DE SOPORTE
# ============================================================

class BacktestDataProvider(DataProvider):
    """Proveedor de datos para backtest con datos históricos"""
    
    def __init__(self, events_queue, symbol_list, historical_data):
        """
        historical_data: dict {symbol: DataFrame con OHLCV}
        """
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        
        # Datos históricos indexados
        self.historical_data = historical_data
        self.latest_data = {s: [] for s in symbol_list}
        self.latest_data_5m = {s: [] for s in symbol_list}
        self.latest_data_15m = {s: [] for s in symbol_list}
        self.latest_data_1h = {s: [] for s in symbol_list}
        
        self.current_index = 0
        self.continue_backtest = True
        
    def get_latest_bars(self, symbol, n=1):
        """Retorna las últimas n barras"""
        try:
            bars_list = self.latest_data[symbol]
            return bars_list[-n:] if len(bars_list) >= n else bars_list
        except KeyError:
            return []
    
    def get_latest_bars_5m(self, symbol, n=1):
        return self.latest_data_5m.get(symbol, [])[-n:]
    
    def get_latest_bars_15m(self, symbol, n=1):
        return self.latest_data_15m.get(symbol, [])[-n:]
    
    def get_latest_bars_1h(self, symbol, n=1):
        return self.latest_data_1h.get(symbol, [])[-n:]
    
    def update_bars(self):
        """Avanza una barra en el tiempo"""
        for symbol in self.symbol_list:
            df = self.historical_data.get(symbol)
            if df is None or self.current_index >= len(df):
                self.continue_backtest = False
                return
            
            row = df.iloc[self.current_index]
            bar = {
                'symbol': symbol,
                'datetime': row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            
            self.latest_data[symbol].append(bar)
            
            # Agregar a timeframes mayores cada N barras
            if self.current_index % 5 == 0:
                self.latest_data_5m[symbol].append(bar)
            if self.current_index % 15 == 0:
                self.latest_data_15m[symbol].append(bar)
            if self.current_index % 60 == 0:
                self.latest_data_1h[symbol].append(bar)
        
        self.current_index += 1
        self.events_queue.put(MarketEvent())


class BacktestPortfolio:
    """Portfolio simplificado para backtest"""
    
    def __init__(self, initial_capital=100.0, leverage=10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        
        # Tracking
        self.positions = {}  # {symbol: {'qty': N, 'entry': P, 'side': 'LONG'/'SHORT'}}
        self.trades = []  # Lista de trades completados
        self.equity_curve = [initial_capital]
        self.timestamps = []
        
        # Métricas
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def get_total_equity(self):
        return self.current_capital
    
    def open_position(self, symbol, side, price, size_usd, timestamp, sl_price=None, tp_price=None):
        """Abre una posición"""
        if symbol in self.positions:
            return False  # Ya hay posición abierta
        
        # Calcular cantidad
        qty = (size_usd * self.leverage) / price
        
        # Comisión de entrada
        commission = size_usd * COMMISSION_PCT
        self.current_capital -= commission
        
        self.positions[symbol] = {
            'qty': qty,
            'entry': price,
            'side': side,
            'size_usd': size_usd,
            'timestamp': timestamp,
            'metadata': None,
            'sl_price': sl_price,
            'tp_price': tp_price
        }
        return True
    
    def open_position_with_metadata(self, symbol, side, price, size_usd, timestamp, metadata=None, sl_price=None, tp_price=None):
        """Abre posición con metadatos (ATR, Regime)"""
        if self.open_position(symbol, side, price, size_usd, timestamp, sl_price, tp_price):
            if metadata:
                self.positions[symbol]['metadata'] = metadata
            return True
        return False
    
    def close_position(self, symbol, price, timestamp):
        """Cierra una posición existente"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        qty = pos['qty']
        entry = pos['entry']
        side = pos['side']
        size_usd = pos['size_usd']
        
        # Calcular PnL
        if side == 'LONG':
            pnl_pct = (price - entry) / entry
        else:  # SHORT
            pnl_pct = (entry - price) / entry
        
        pnl_usd = size_usd * self.leverage * pnl_pct
        
        # Comisión de salida
        commission = size_usd * COMMISSION_PCT
        pnl_usd -= commission
        
        self.current_capital += pnl_usd
        
        # Registrar trade
        trade = {
            'symbol': symbol,
            'side': side,
            'entry': entry,
            'exit': price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'entry_time': pos['timestamp'],
            'exit_time': timestamp,
            'duration': (timestamp - pos['timestamp']).total_seconds() / 60 if isinstance(timestamp, datetime) else 0,
            'metadata': pos.get('metadata', {})
        }
        self.trades.append(trade)
        
        # Actualizar métricas
        if pnl_usd > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Actualizar drawdown
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
        current_dd = (self.peak_equity - self.current_capital) / self.peak_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        del self.positions[symbol]
        return trade
    
    def update_equity(self, timestamp):
        """Actualiza curva de equity"""
        self.equity_curve.append(self.current_capital)
        if isinstance(timestamp, datetime):
            self.timestamps.append(timestamp)


# ============================================================
# FUNCIONES DE BACKTEST
# ============================================================

def fetch_binance_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    Descarga datos históricos de Binance
    
    QUÉ: Función para obtener velas 1m
    CÓMO: Usa python-binance REST API (siempre mainnet para datos históricos)
    NOTA: Usamos mainnet porque testnet tiene datos históricos limitados
    """
    print(f"📡 Descargando {days} días de datos para {symbol}...")
    
    # SIEMPRE usar mainnet para datos históricos (es solo lectura, seguro)
    # Testnet tiene muy pocos datos históricos
    client = Client()  # Sin API keys = solo datos públicos

    
    # Calcular rango
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Convertir símbolo
    binance_symbol = symbol.replace('/', '')
    
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        # Binance limita a 1000 velas por request
        klines = client.get_historical_klines(
            binance_symbol,
            Client.KLINE_INTERVAL_1MINUTE,
            str(int(current_start.timestamp() * 1000)),
            str(int(min(current_start + timedelta(hours=16), end_time).timestamp() * 1000)),
            limit=1000
        )
        
        if not klines:
            break
            
        all_klines.extend(klines)
        current_start += timedelta(hours=16)
        time.sleep(0.2)  # Rate limit
    
    # Convertir a DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"✅ Descargados {len(df)} velas ({len(df)/60/24:.1f} días)")
    return df


def run_backtest(data: pd.DataFrame, symbol: str = 'BTC/USDT') -> dict:
    """
    Ejecuta el backtest completo
    
    QUÉ: Simulación tick-by-tick de la estrategia
    CÓMO: Itera por cada vela, genera señales, ejecuta trades
    """
    print("\n🚀 Iniciando backtest...")
    
    events_queue = Queue()
    historical_data = {symbol: data}
    
    # Inicializar componentes
    data_provider = BacktestDataProvider(events_queue, [symbol], historical_data)
    portfolio = BacktestPortfolio(INITIAL_CAPITAL, LEVERAGE)
    strategy = HybridScalpingStrategy(data_provider, events_queue)
    
    # Variables de control
    warmup_bars = 100  # Barras para calentar indicadores
    signals_generated = 0
    trades_executed = 0
    last_signal_idx = -50  # Evitar señales muy seguidas
    
    # Stop Loss / Take Profit tracking
    active_sl = None
    active_tp = None
    
    bar_count = 0
    total_bars = len(data)
    
    print(f"📊 Procesando {total_bars} barras...")
    
    while data_provider.continue_backtest:
        # Actualizar datos
        data_provider.update_bars()
        bar_count += 1
        
        if bar_count < warmup_bars:
            continue
        
        # Obtener precio actual
        bars = data_provider.get_latest_bars(symbol, 1)
        if not bars:
            continue
        
        current_bar = bars[-1]
        current_price = current_bar['close']
        current_time = current_bar['datetime']
        high = current_bar['high']
        low = current_bar['low']
        
        # Verificar SL/TP para posiciones abiertas
        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            entry = pos['entry']
            side = pos['side']
            
            # Calcular trailing stop basado en ATR (simplificado)
            atr_approx = (high - low) * 2  # Aproximación simple
            
            # Check Exit Conditions (using Stored SL/TP)
            # If SL/TP not stored (legacy), fallback or ignore
            stored_sl = pos.get('sl_price')
            stored_tp = pos.get('tp_price')
            
            # Default fallbacks if None (legacy safety)
            if stored_sl is None:
                if side == 'LONG': stored_sl = entry * 0.985
                else: stored_sl = entry * 1.015
            if stored_tp is None:
                if side == 'LONG': stored_tp = entry * 1.01
                else: stored_tp = entry * 0.99

            if side == 'LONG':
                if low <= stored_sl:
                    trade = portfolio.close_position(symbol, stored_sl, current_time) # Execute at SL
                    if trade: trades_executed += 1
                elif high >= stored_tp:
                    trade = portfolio.close_position(symbol, stored_tp, current_time) # Execute at TP
                    if trade: trades_executed += 1
            else: # SHORT
                if high >= stored_sl:
                    trade = portfolio.close_position(symbol, stored_sl, current_time)
                    if trade: trades_executed += 1
                elif low <= stored_tp:
                    trade = portfolio.close_position(symbol, stored_tp, current_time)
                    if trade: trades_executed += 1
        
        # Generar señales solo si no hay posición y han pasado suficientes barras
        if symbol not in portfolio.positions and (bar_count - last_signal_idx) > 30:
            # Usar estrategia simplificada para backtest
            all_bars = data_provider.get_latest_bars(symbol, 100)
            signal_type, strength = calculate_simple_signal(all_bars)
            
            if signal_type is not None and strength >= 0.6:
                signals_generated += 1
                last_signal_idx = bar_count
                
                # Metadata capture
                atr_val = 0.0
                try:
                    # Reconstruir ATR correcto (14 period)
                    if len(all_bars) >= 15:
                        closes = np.array([b['close'] for b in all_bars[-15:]])
                        highs = np.array([b['high'] for b in all_bars[-15:]])
                        lows = np.array([b['low'] for b in all_bars[-15:]])
                        # Simple ATR calc
                        tr_list = []
                        for i in range(1, len(closes)):
                            hl = highs[i] - lows[i]
                            hc = abs(highs[i] - closes[i-1])
                            lc = abs(lows[i] - closes[i-1])
                            tr_list.append(max(hl, hc, lc))
                        atr_val = sum(tr_list[-14:]) / 14 if len(tr_list) >= 14 else tr_list[-1]
                except:
                    atr_val = current_price * 0.01

                regime = "RANGING"
                if atr_val / current_price > 0.01: # High Vol
                    regime = "VOLATILE"
                
                metadata = {
                    'atr': float(atr_val),
                    'regime': regime
                }

                # === DYNAMIC RISK LOGIC (Mirroring RiskManager) ===
                # 1. Base Logic (Drawdown Protection)
                peak = portfolio.peak_equity
                current_cap = portfolio.current_capital
                initial = portfolio.initial_capital
                
                dd = (peak - current_cap) / peak if peak > 0 else 0
                
                risk_pct = 0.01 # Default 1%
                if dd > 0.10:
                    risk_pct = 0.005 # 0.5%
                elif dd > 0.05:
                    risk_pct = 0.0075 # 0.75%

                # 2. Profit Lock Milestones (Wealth Preservation)
                if peak >= (initial * 2.0): # +100% Growth
                    risk_pct *= 0.10 # Reduce to 10% (0.1% risk)
                elif peak >= (initial * 1.5): # +50% Growth
                    risk_pct *= 0.25 # Reduce to 25% (0.25% risk)
                    
                # 3. Protected Capital Floor (The Ratchet)
                profit = peak - initial
                if profit > 0:
                    protected_capital = initial + (profit * 0.80)
                    max_loss_allowed = current_cap - protected_capital
                    
                    if max_loss_allowed <= 0:
                         risk_pct = 0.0 # Stop Trading
                    else:
                         current_risk_amt = current_cap * risk_pct
                         if current_risk_amt > max_loss_allowed:
                             risk_pct = max_loss_allowed / current_cap
                
                risk_usd = current_cap * risk_pct
                
                # 2. Dynamic SL based on ATR metadata
                # (Re-using atr_val calculated above)
                atr_pct = atr_val / current_price
                if atr_pct < 0.005:  # Low Vol
                    sl_mult = 3.0
                elif atr_pct > 0.01: # High Vol
                    sl_mult = 2.0
                else:
                    sl_mult = 2.5
                
                sl_pct = max(0.003, min(atr_pct * sl_mult, 0.015))
                
                # 3. Volatility Based Sizing
                # Size = Risk / SL_Pct
                size_usd = risk_usd / sl_pct
                
                # Cap size (e.g. max 50% of account for safety or safe leverage)
                max_lev = 10
                if atr_pct > 0.02: max_lev = 3
                elif atr_pct > 0.01: max_lev = 5
                elif atr_pct > 0.005: max_lev = 8
                
                max_size = current_cap * max_lev
                size_usd = min(size_usd, max_size)
                
                risk_capital = size_usd # Used for open_position

                # Ejecutar trade
                side = 'LONG' if signal_type == SignalType.LONG else 'SHORT'
                # Calc Entry SL/TP
                tp_pct = max(0.005, sl_pct * 1.5) # 1.5 R:R
                
                # MICRO-SCALPING SIZE BOOST (Corrected)
                raw_notional_usd = risk_usd / sl_pct if sl_pct > 0 else 0
                notional_usd = max(raw_notional_usd, 5.0) # Boost to Min $5 Notional
                
                # Calculate Margin Required
                margin_required = notional_usd / portfolio.leverage
                
                # Check Collateral
                if margin_required > portfolio.current_capital:
                    margin_required = portfolio.current_capital * 0.99 # Leave 1% buffer
                    
                size_usd = margin_required # Pass Margin to open_position
                    
                
                if side == 'LONG':
                    entry_sl = current_price * (1 - sl_pct)
                    entry_tp = current_price * (1 + tp_pct)
                else: # SHORT
                    entry_sl = current_price * (1 + sl_pct)
                    entry_tp = current_price * (1 - tp_pct)

                opened = portfolio.open_position_with_metadata(
                    symbol, side, current_price, risk_capital, current_time, metadata, entry_sl, entry_tp
                )
                
                if opened:
                    trades_executed += 1
                    if trades_executed <= 10 or trades_executed % 20 == 0:
                        print(f"  🎯 Trade #{trades_executed}: {side} @ ${current_price:.2f}")

        
        # Actualizar equity cada hora
        if bar_count % 60 == 0:
            portfolio.update_equity(current_time)
        
        # Progreso
        if bar_count % 5000 == 0:
            progress = bar_count / total_bars * 100
            print(f"  ▸ {progress:.1f}% completado ({bar_count}/{total_bars})")
    
    # Cerrar posiciones abiertas al final
    for symbol in list(portfolio.positions.keys()):
        bars = data_provider.get_latest_bars(symbol, 1)
        if bars:
            portfolio.close_position(symbol, bars[-1]['close'], bars[-1]['datetime'])
            trades_executed += 1
    
    print(f"\n✅ Backtest completado: {trades_executed} trades ejecutados")
    
    return {
        'portfolio': portfolio,
        'signals': signals_generated,
        'trades': trades_executed,
        'bars_processed': bar_count
    }


def calculate_metrics(portfolio: BacktestPortfolio) -> dict:
    """
    Calcula métricas de rendimiento
    
    QUÉ: Sharpe, Drawdown, Win Rate, etc.
    POR QUÉ: Validar targets antes de producción
    """
    print("\n📈 Calculando métricas...")
    
    trades = portfolio.trades
    equity_curve = portfolio.equity_curve
    
    if not trades:
        return {
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_trade_pnl_usd': 0,
            'avg_trade_pnl_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'profit_factor': 0,
            'avg_trade_duration_min': 0,
            'final_capital': portfolio.current_capital,
            'peak_capital': portfolio.peak_equity
        }
    
    # Returns diarios (aproximado por equity curve)
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # Sharpe Ratio (anualizado, asumiendo 365 días trading)
    if len(returns) > 1 and np.std(returns) > 0:
        # Escalar a daily returns (cada punto = ~1 hora)
        daily_returns = []
        for i in range(0, len(returns), 24):
            chunk = returns[i:i+24]
            if len(chunk) > 0:
                daily_returns.append(np.sum(chunk))
        
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    # Max Drawdown
    max_dd = portfolio.max_drawdown * 100
    
    # Win Rate
    total_trades = len(trades)
    winning = portfolio.winning_trades
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    
    # Total Return
    final_capital = portfolio.current_capital
    initial = portfolio.initial_capital
    total_return = ((final_capital - initial) / initial) * 100
    
    # Average Trade PnL
    avg_pnl = np.mean([t['pnl_usd'] for t in trades]) if trades else 0
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
    
    # Profit Factor
    gross_profit = sum([t['pnl_usd'] for t in trades if t['pnl_usd'] > 0])
    gross_loss = abs(sum([t['pnl_usd'] for t in trades if t['pnl_usd'] < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Avg trade duration
    avg_duration = np.mean([t['duration'] for t in trades]) if trades else 0
    
    return {
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'win_rate': win_rate,
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': winning,
        'losing_trades': total_trades - winning,
        'avg_trade_pnl_usd': avg_pnl,
        'avg_trade_pnl_pct': avg_pnl_pct,
        'profit_factor': profit_factor,
        'avg_trade_duration_min': avg_duration,
        'final_capital': final_capital,
        'peak_capital': portfolio.peak_equity
    }


def print_report(metrics: dict, portfolio: BacktestPortfolio):
    """
    Imprime reporte detallado usando método PROFESOR
    """
    print("\n" + "="*60)
    print("📊 REPORTE DE BACKTEST - TRADER GEMINI")
    print("="*60)
    
    # Métricas principales
    sharpe = metrics['sharpe_ratio']
    max_dd = metrics['max_drawdown_pct']
    win_rate = metrics['win_rate']
    total_return = metrics['total_return']
    
    print(f"\n🎯 MÉTRICAS PRINCIPALES:")
    print(f"   Sharpe Ratio:     {sharpe:>10.2f}  {'✅' if sharpe > 2.0 else '⚠️'} (Target: > 2.0)")
    print(f"   Max Drawdown:     {max_dd:>10.2f}% {'✅' if max_dd < 1.5 else '⚠️'} (Target: < 1.5%)")
    print(f"   Win Rate:         {win_rate:>10.1f}%")
    print(f"   Total Return:     {total_return:>10.2f}%")
    
    print(f"\n📈 ESTADÍSTICAS DE TRADING:")
    print(f"   Total Trades:     {metrics['total_trades']:>10}")
    print(f"   Winning Trades:   {metrics['winning_trades']:>10}")
    print(f"   Losing Trades:    {metrics['losing_trades']:>10}")
    print(f"   Avg Trade PnL:    ${metrics['avg_trade_pnl_usd']:>9.2f}")
    print(f"   Profit Factor:    {metrics['profit_factor']:>10.2f}")
    print(f"   Avg Duration:     {metrics['avg_trade_duration_min']:>10.1f} min")
    
    print(f"\n💰 CAPITAL:")
    print(f"   Initial:          ${portfolio.initial_capital:>9.2f}")
    print(f"   Final:            ${metrics['final_capital']:>9.2f}")
    print(f"   Peak:             ${metrics['peak_capital']:>9.2f}")
    
    # Análisis PROFESOR
    print("\n" + "="*60)
    print("👨‍🏫 ANÁLISIS MÉTODO PROFESOR")
    print("="*60)
    
    print("\n📌 QUÉ: Resultados del backtest de 1 mes")
    print(f"   → Se ejecutaron {metrics['total_trades']} trades simulados")
    print(f"   → Capital final: ${metrics['final_capital']:.2f} ({total_return:+.2f}%)")
    
    print("\n📌 POR QUÉ estos resultados:")
    if sharpe > 2.0:
        print("   → Sharpe alto indica buen ratio retorno/riesgo")
    else:
        print("   → Sharpe bajo sugiere volatilidad o retornos insuficientes")
    
    if max_dd < 1.5:
        print("   → Drawdown controlado muestra buena gestión de riesgo")
    else:
        print("   → Drawdown alto indica exposición excesiva o SL amplios")
    
    print("\n📌 PARA QUÉ sirven estas métricas:")
    print("   → Sharpe: Evaluar si el retorno justifica el riesgo")
    print("   → Drawdown: Medir máxima pérdida desde pico")
    print("   → Win Rate: Probabilidad de trade ganador")
    
    print("\n📌 CÓMO se calcularon:")
    print("   → Sharpe = (Returns promedio / Std Returns) × √365")
    print("   → Drawdown = (Peak - Current) / Peak × 100")
    print("   → Win Rate = Trades ganadores / Total trades × 100")
    
    # Veredicto final
    print("\n" + "="*60)
    passed = sharpe > 2.0 and max_dd < 1.5
    if passed:
        print("🟢 VEREDICTO: APROBADO - Sistema listo para producción")
    else:
        print("🟡 VEREDICTO: REVISIÓN NECESARIA")
        if sharpe <= 2.0:
            print("   ⚠️ Mejorar Sharpe: Ajustar TP/SL o filtros de entrada")
        if max_dd >= 1.5:
            print("   ⚠️ Reducir Drawdown: Reducir tamaño posición o leverage")
    print("="*60)
    
    return passed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🧪 BACKTEST TRADER GEMINI - FULL BASKET (26 SYMBOLS)")
    print("="*60)
    
    try:
        # Load Symbols (Full Smart Basket)
        symbols = Config.CRYPTO_FUTURES_PAIRS
        print(f"📋 Testing Full Smart Basket ({len(symbols)} symbols)...")
        
        grand_total_trades = 0
        grand_winning_trades = 0
        grand_losing_trades = 0
        grand_pnl_usd = 0.0
        
        # Aggregate Portfolio mimicking single account
        # We simulate "Parallel" processing by adding up PnL, assuming capital is shared or allocated
        # For simplicity, we track PnL summation on top of Initial Capital
        
        all_results = []
        
        # Use simple global portfolio for aggregation
        # Note: BacktestPortfolio logic is single-threaded/sequential here, 
        # so we will sum up the PnL impacts.
        
        print(f"\n💰 STARTING CAPITAL: ${INITIAL_CAPITAL:.2f}")
        current_equity = INITIAL_CAPITAL
        
        for i, symbol in enumerate(symbols):
            print(f"\n🔹 TESTING {symbol} ({i+1}/{len(symbols)})...")
            
            # Rate Limit Protection
            time.sleep(2) # 2s delay between symbols to avoid ban
            
            # 1. Download Data
            try:
                data = fetch_binance_data(symbol, days=DAYS)
            except Exception as e:
                print(f"   ⚠️ Download failed for {symbol}: {e}")
                continue
            
            if data.empty:
                print(f"   ⚠️ No data for {symbol}, skipping.")
                continue
                
            # 2. Run Backtest
            # Reset portfolio for each symbol to isolate logic per pair (then aggregate PnL)
            # OR share portfolio? Shared is harder to mock sequentially.
            # We will use ISOLATED logic per pair and sum PnL.
            results = run_backtest(data, symbol)
            
            # 3. Aggregate
            p = results['portfolio']
            symbol_pnl = p.current_capital - p.initial_capital
            
            grand_pnl_usd += symbol_pnl
            grand_total_trades += len(p.trades)
            grand_winning_trades += p.winning_trades
            grand_losing_trades += p.losing_trades
            
            print(f"   👉 Result {symbol}: ${symbol_pnl:+.2f} ({len(p.trades)} trades)")
            
            all_results.append({
                'symbol': symbol,
                'pnl': symbol_pnl,
                'trades': len(p.trades),
                'wins': p.winning_trades
            })
            
        # Final Totals
        final_capital = INITIAL_CAPITAL + grand_pnl_usd
        total_return_pct = (grand_pnl_usd / INITIAL_CAPITAL) * 100
        total_win_rate = (grand_winning_trades / grand_total_trades * 100) if grand_total_trades > 0 else 0
        
        print("\n" + "="*60)
        print("🏆 GRAND TOTAL REPORT (26 SYMBOLS)")
        print("="*60)
        print(f"💰 Initial Capital: ${INITIAL_CAPITAL:.2f}")
        print(f"💰 Final Capital:   ${final_capital:.2f}")
        print(f"📈 Total PnL:       ${grand_pnl_usd:+.2f} ({total_return_pct:+.2f}%)")
        print(f"📊 Total Trades:    {grand_total_trades}")
        print(f"✅ Win Rate:        {total_win_rate:.1f}%")
        
        print("\n🏅 TOP PERFORMERS:")
        sorted_results = sorted(all_results, key=lambda x: x['pnl'], reverse=True)
        for r in sorted_results[:5]:
            print(f"   1. {r['symbol']}: ${r['pnl']:+.2f}")
            
        print("\n💀 WORST PERFORMERS:")
        for r in sorted_results[-5:]:
            print(f"   - {r['symbol']}: ${r['pnl']:+.2f}")
            
        # Save JSON
        output_file = 'backtest_smart_full_results.json'
        import json
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'initial_capital': INITIAL_CAPITAL,
                'final_capital': final_capital,
                'total_pnl': grand_pnl_usd,
                'total_trades': grand_total_trades,
                'details': all_results
            }, f, indent=2, default=str)
        print(f"\n📁 Full results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Final Execution Error: {e}")
        import traceback
        traceback.print_exc()
