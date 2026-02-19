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
from risk.risk_manager import RiskManager
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import SignalType, OrderSide, EventType
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
        
        self.historical_data = historical_data
        
        # Pre-allocate structured arrays for Zero-Copy parity
        self.struct_data = {s: {} for s in symbol_list}
        struct_dtype = [
            ('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'), 
            ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')
        ]
        
        for s in symbol_list:
            # Main 1m data
            df_1m = historical_data[s]
            self.struct_data[s]['1m'] = self._df_to_struct(df_1m, struct_dtype)
            
            # Resampled data
            for tf in ['5min', '15min', '1h']:
                df_res = df_1m.resample(tf).agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                key = tf.lower().replace('min', 'm').replace('h', 'h')
                self.struct_data[s][key] = self._df_to_struct(df_res, struct_dtype)
        
        self.current_index = 0
        self.current_time_ms = 0
        self.continue_backtest = True

    def _df_to_struct(self, df, dtype):
        """Converts DataFrame to NumPy Structured Array efficiently"""
        res = np.empty(len(df), dtype=dtype)
        res['timestamp'] = df.index.values.astype('datetime64[ms]').astype('int64')
        res['open'] = df['open'].values
        res['high'] = df['high'].values
        res['low'] = df['low'].values
        res['close'] = df['close'].values
        res['volume'] = df['volume'].values
        return res
        
    def get_latest_bars(self, symbol, n=1, timeframe='1m'):
        """Retorna vista de arreglo estructurado (Ultra-Fast slicing)"""
        try:
            arr = self.struct_data[symbol][timeframe]
            
            # Find index of current_time_ms in this timeframe
            # Using searchsorted (returns index where element should be inserted to maintain order)
            # side='right' finds the first index > current_time_ms
            idx = np.searchsorted(arr['timestamp'], self.current_time_ms, side='right')
            
            if idx == 0: return None
            
            start = max(0, idx - n)
            return arr[start:idx] # Returns a view (Zero-Copy)
        except Exception:
            return None

    def get_active_positions(self):
        """Mock for strategy compatibility"""
        return {}
        
    def get_symbol_precision(self, symbol):
        """Mock for strategy compatibility"""
        return {'quantity': 3, 'price': 2}
    
    def get_latest_bars_5m(self, symbol, n=1):
        return self.get_latest_bars(symbol, n, '5m')
    
    def get_latest_bars_15m(self, symbol, n=1):
        return self.get_latest_bars(symbol, n, '15m')
    
    def get_latest_bars_1h(self, symbol, n=1):
        return self.get_latest_bars(symbol, n, '1h')
    
    def update_bars(self):
        """Avanza una barra en el tiempo"""
        symbol = self.symbol_list[0]
        arr = self.struct_data[symbol]['1m']
        
        if self.current_index >= len(arr):
            self.continue_backtest = False
            return
            
        self.current_time_ms = arr['timestamp'][self.current_index]
        close_price = arr['close'][self.current_index]
        
        self.current_index += 1
        
        # Dispatch event
        self.events_queue.put(MarketEvent(
            symbol=symbol, 
            close_price=close_price, 
            timestamp=pd.to_datetime(self.current_time_ms, unit='ms', utc=True)
        ))


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
    
    def _apply_slippage(self, price, side):
        """Simulate realistic slippage based on volatility"""
        # Base slippage: 0.01% to 0.05%
        import random
        slip_pct = random.uniform(0.0001, 0.0005)
        
        # Apply against direction
        if side == 'LONG':
            return price * (1 + slip_pct) # Buy higher
        else:
            return price * (1 - slip_pct) # Sell lower

    def open_position(self, symbol, side, price, size_usd, timestamp, sl_price=None, tp_price=None):
        """Abre una posición con Slippage Simulado"""
        if symbol in self.positions:
            return False  # Ya hay posición abierta
        
        # Apply Slippage
        filled_price = self._apply_slippage(price, side)
        
        # Calcular cantidad
        qty = (size_usd * self.leverage) / filled_price
        
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
        
        # Apply Slippage to Exit
        exit_side = 'SHORT' if side == 'LONG' else 'LONG'
        filled_price = self._apply_slippage(price, exit_side)
        
        # Calcular PnL
        if side == 'LONG':
            pnl_pct = (filled_price - entry) / entry
        else:  # SHORT
            pnl_pct = (entry - filled_price) / entry
        
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
        current_time = pd.to_datetime(current_bar['timestamp'], unit='ms', utc=True)
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
        
        # 3. GENERATE SIGNALS (SUPREMO-V3 Real Logic)
        # Sync strategy state with portfolio (for signal generation logic)
        strategy.bought[symbol] = symbol in portfolio.positions
        
        # Call strategy every bar to allow EXIT signals and state updates
        market_event = MarketEvent(symbol=symbol, close_price=current_price, timestamp=current_time)
        # Note: We need to set the data handler in the strategy state if it uses it directly
        # but HybridScalpingStrategy uses self.data_provider passed in constructor.
        strategy.calculate_signals(market_event)
        
        # Process signals from queue
        while not events_queue.empty():
            event = events_queue.get()
            if not isinstance(event, SignalEvent):
                continue
            
            # Handle EXIT signals
            if event.signal_type == SignalType.EXIT:
                if symbol in portfolio.positions:
                    trade = portfolio.close_position(symbol, current_price, current_time)
                    if trade: trades_executed += 1
                continue

            # Handle ENTRY signals
            if symbol not in portfolio.positions:
                signals_generated += 1
                last_signal_idx = bar_count
                
                # Metadata capture (Aligned with RiskManager)
                meta_dict = event.metadata if event.metadata else {}
                metadata = {
                    'atr': getattr(event, 'atr', 0.0),
                    'confluence': meta_dict.get('multi_timeframe_score', 0.0)
                }

                # === DYNAMIC RISK & SIZING (Aligned with Supremo-V3) ===
                # 1. Base Logic (Drawdown Protection)
                peak = portfolio.peak_equity
                current_cap = portfolio.current_capital
                initial = portfolio.initial_capital
                
                dd = (peak - current_cap) / peak if peak > 0 else 0
                
                risk_pct = 0.01 # Standard Institutional Risk (1%)
                if dd > 0.05: risk_pct = 0.02 
                if dd > 0.10: risk_pct = 0.01 

                # 2. Profit Lock Milestones
                if peak >= (initial * 2.0): risk_pct *= 0.50 
                elif peak >= (initial * 1.5): risk_pct *= 0.75 
               
                risk_usd = current_cap * risk_pct
                
                # 3. Position Sizing based on SL from Signal
                # FIXED: Handles both percentage (2.0) and decimal (0.02)
                raw_sl_pct = getattr(event, 'sl_pct', 1.5)
                sl_decimal = raw_sl_pct / 100.0 if raw_sl_pct > 0.1 else raw_sl_pct
                tp_pct = getattr(event, 'tp_pct', 2.0)
                
                # Size = Risk / SL_Pct
                size_usd = (risk_usd / sl_decimal) if sl_decimal > 0 else (current_cap * 0.1)
                
                # Hard cap sizing (prevent extreme leverage)
                max_size = current_cap * 10 
                size_usd = min(size_usd, max_size)
                
                # Institutional Minimum
                if size_usd < 5.0:
                    continue

                # 4. EXECUTE TRADE (Instant Fill with Slippage)
                side = 'LONG' if event.signal_type == SignalType.LONG else 'SHORT'
                
                # FIXED: Standardizing decimal usage for entry calculations
                tp_decimal = tp_pct / 100.0 if tp_pct > 0.1 else tp_pct
                
                if side == 'LONG':
                    entry_sl = current_price * (1 - sl_decimal)
                    entry_tp = current_price * (1 + tp_decimal)
                else: # SHORT
                    entry_sl = current_price * (1 + sl_decimal)
                    entry_tp = current_price * (1 - tp_decimal)

                opened = portfolio.open_position_with_metadata(
                    symbol, side, current_price, size_usd, current_time, metadata, entry_sl, entry_tp
                )
                
                if opened:
                    trades_executed += 1
                    if trades_executed <= 10 or trades_executed % 20 == 0:
                        print(f"  🎯 Trade #{trades_executed}: {side} @ ${current_price:.2f} (SL: {sl_decimal*100:.2f}%, TP: {tp_decimal*100:.2f}%)")

        
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
        if bars is not None and len(bars) > 0:
            ts_ms = bars[-1]['timestamp']
            dt_close = pd.to_datetime(ts_ms, unit='ms', utc=True)
            portfolio.close_position(symbol, bars[-1]['close'], dt_close)
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
        'initial_capital': initial,
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
        # Load Symbols (Full Smart Basket - 20 symbols for Profitability Audit)
        symbols = Config.CRYPTO_FUTURES_PAIRS[:20]
        print(f"📋 Testing Aggressive Optimization Subset ({len(symbols)} symbols)...")
        
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
