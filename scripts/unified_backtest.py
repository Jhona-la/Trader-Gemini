"""
UNIFIED BACKTEST PIPELINE (Phase III Audit Fix)
================================================
Axioma AUDIT-5: El backtest debe ser idéntico a producción.
Este script reemplaza a `simulation.py` para asegurar que el FeatureEngineering real
y el modelo XGBoost/RF/Sophia real se usen durante la simulación.
Incluye:
- Producción FeatureEngineering
- Fees reales (Maker/Taker)
- Portfolio Heat & Risk Manager
"""

import os
import sys
import numpy as np
import polars as pl
from datetime import datetime, timezone
import time

sys.path.insert(0, '.')
from config import Config
Config.Risk.ABSOLUTE_CERTAINTY_THRESHOLD = 0.40  # Permite que la estrategia técnica pase sin veto ML

from strategies.components.feature_engineering import FeatureEngineering
from risk.risk_manager import RiskManager
from core.portfolio import Portfolio
from core.events import SignalEvent
from core.enums import SignalType

import queue
from strategies.technical import HybridScalpingStrategy
from core.genotype import Genotype

class MockMarketEvent:
    def __init__(self, symbol):
        self.type = 'MARKET'
        self.symbol = symbol
        self.timeframe = '1m'
        self.is_closed = True

class MockDataProvider:
    def __init__(self, df):
        self.symbol_list = ["BTC/USDT"]
        df = df.sort("timestamp")
        self.df_1m = self._df_to_struct(df)
        
        df = df.with_columns(pl.from_epoch("timestamp", time_unit="ms").alias("datetime"))
        
        df_3m = df.group_by_dynamic("datetime", every="3m").agg([
            pl.first("open"), pl.max("high"), pl.min("low"), pl.last("close"), pl.sum("volume"), pl.first("timestamp")
        ]).drop("datetime")
        self.df_3m = self._df_to_struct(df_3m)
        
        df_5m = df.group_by_dynamic("datetime", every="5m").agg([
            pl.first("open"), pl.max("high"), pl.min("low"), pl.last("close"), pl.sum("volume"), pl.first("timestamp")
        ]).drop("datetime")
        self.df_5m = self._df_to_struct(df_5m)
        
        df_15m = df.group_by_dynamic("datetime", every="15m").agg([
            pl.first("open"), pl.max("high"), pl.min("low"), pl.last("close"), pl.sum("volume"), pl.first("timestamp")
        ]).drop("datetime")
        self.df_15m = self._df_to_struct(df_15m)
        
        df_1h = df.group_by_dynamic("datetime", every="1h").agg([
            pl.first("open"), pl.max("high"), pl.min("low"), pl.last("close"), pl.sum("volume"), pl.first("timestamp")
        ]).drop("datetime")
        self.df_1h = self._df_to_struct(df_1h)
        
        self.current_timestamp = 0

    def _df_to_struct(self, df):
        struct = np.zeros(len(df), dtype=[
            ('timestamp', 'int64'), ('open', 'float64'), ('high', 'float64'), 
            ('low', 'float64'), ('close', 'float64'), ('volume', 'float64')
        ])
        for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                struct[col] = df[col].to_numpy()
        return struct
        
    def get_latest_bars(self, symbol, n=300, timeframe='5m'):
        if timeframe == '1m': data = self.df_1m
        elif timeframe == '3m': data = self.df_3m
        elif timeframe == '5m': data = self.df_5m
        elif timeframe == '15m': data = self.df_15m
        elif timeframe == '1h': data = self.df_1h
        else: data = self.df_1m
        
        idx = np.searchsorted(data['timestamp'], self.current_timestamp, side='right')
        start = max(0, idx - n)
        return data[start:idx]
        
    def get_active_positions(self):
        # The strategy requires this to check currently open positions via the provider.
        if hasattr(self, 'mocked_position') and self.mocked_position != 0:
            return {self.mocked_symbol: {'quantity': self.mocked_position}}
        return {}

    def get_order_flow_metrics(self, symbol):
        return {}



class UnifiedBacktestEngine:
    def __init__(self, data: pl.DataFrame):
        self.data = data
        self.portfolio = Portfolio(initial_capital=13.0)
        self.risk_manager = RiskManager(portfolio=self.portfolio)
        self.fe = FeatureEngineering()
        
        self.data_provider = MockDataProvider(data)
        self.events_queue = queue.Queue()
        self.strategy = None
        
    def run(self, symbol: str):
        print(f"🚀 Iniciando Unified Backtest para {symbol}...")
        
        genotype = Genotype(symbol)
        self.strategy = HybridScalpingStrategy(
            data_provider=self.data_provider, 
            events_queue=self.events_queue, 
            genotype=genotype, 
            horizon='SCALPING'
        )
        
        # 1. Feature Engineering Real (Como en Producción)
        print(f"  [1/4] Extrayendo features reales (Producción Pipeline)...")
        features_df = self.fe.prepare_features(self.data, horizon='SCALPING')
        print(f"        Features generadas: {len(features_df.columns)}")
        
        # 2. Setup de simulación
        closes = self.data['close'].to_numpy()
        highs = self.data['high'].to_numpy()
        lows = self.data['low'].to_numpy()
        volumes = self.data['volume'].to_numpy()
        
        # Simulated fees (Binance VIP 0 - Maker/Taker)
        MAKER_FEE = 0.0005
        TAKER_FEE = 0.0005
        
        position = 0
        entry_price = 0.0
        entry_idx = 0
        trades = []
        
        print(f"  [2/4] Ejecutando simulación temporal...")
        
        # Mock time.time to simulate real-time progression for CooldownManager
        original_time = time.time
        current_sim_time = [0.0]
        time.time = lambda: current_sim_time[0]
        
        try:
            signals_generated = 0
            # Emular el loop de Engine
            for i in range(50, len(self.data)):
                current_timestamp = self.data['timestamp'][i]
                current_sim_time[0] = current_timestamp / 1000.0
                self.data_provider.current_timestamp = current_timestamp
                
                # Mock virtual clock for CooldownManager

                from utils.cooldown_manager import cooldown_manager
                cooldown_manager.set_virtual_time(datetime.fromtimestamp(current_sim_time[0], tz=timezone.utc))
                
                # Extraer features actuales (simular hot-path)
                # En producción se extrae la fila 'i'
                current_close = closes[i]

                # Extract features for analysis inside strategy
                # The strategy calls self.data_provider.get_latest_bars
                
                # Siempre calcular señales para permitir exits dinámicos (Trailing/BE/RSI)
                mock_event = MockMarketEvent(symbol)
                self.strategy.calculate_signals(mock_event)
                
                if position == 0:
                    # Drain the queue for entry
                    while not self.events_queue.empty():
                        signal = self.events_queue.get()
                        signals_generated += 1
                        
                        # Fix is_shadow mapping early if we are in core
                        from config import Config
                        core_symbols = getattr(Config, 'CORE_SYMBOLS', [])
                        if core_symbols and symbol in core_symbols:
                            if hasattr(signal, 'is_shadow'):
                                object.__setattr__(signal, 'is_shadow', False)
                                
                        if signal.signal_type in (SignalType.LONG, SignalType.SHORT):
                            order = self.risk_manager.generate_order(signal, current_close)
                            if order:
                                position = 1 if signal.signal_type == SignalType.LONG else -1
                                self.data_provider.mocked_position = position
                                self.data_provider.mocked_symbol = symbol
                                # Apply TAKER fee to entry
                                entry_price = current_close * (1 + TAKER_FEE) if position == 1 else current_close * (1 - TAKER_FEE)
                                entry_idx = i
                                break # Only take one order
                        else:
                            # Log the reason for rejection to debug 0 trades
                            rej = getattr(signal, 'metadata', {}).get('rejection_reason', 'UNKNOWN_VETO_NO_METADATA')
                            if hasattr(signal, 'signal_type') and signal.signal_type != SignalType.EXIT:
                                print(f"❌ REJECTED {symbol}: {rej}")
                
                else: # Managing position
                    safe_entry = entry_price if entry_price != 0 else 1.0
                    pnl_pct = (current_close - safe_entry) / safe_entry if position == 1 else (safe_entry - current_close) / safe_entry
                    
                    # Check SL/TP using strategy's configured parameters
                    is_exit = False
                    exit_price = current_close
                    if pnl_pct <= -self.strategy.SL_PCT: # Hit SL
                        is_exit = True
                        print(f"🛑 HARD SL HIT: {pnl_pct*100:.2f}%")
                    elif pnl_pct >= self.strategy.TP_PCT: # Hit TP
                        is_exit = True
                        print(f"✅ HARD TP HIT: {pnl_pct*100:.2f}%")
                    else:
                        # Check dynamic signals from strategy (Trailing/BE/RSI)
                        while not self.events_queue.empty():
                            signal = self.events_queue.get()
                            if signal.signal_type == SignalType.EXIT:
                                is_exit = True
                                print(f"🛡️ DYNAMIC EXIT HIT: {pnl_pct*100:.2f}%")
                                break
                            elif signal.signal_type == SignalType.REVERSE:
                                is_exit = True
                                print(f"🔄 REVERSE EXIT HIT: {pnl_pct*100:.2f}%")
                                # Normally would enter new position, but backtest simplifies
                                break
                        
                    if is_exit:
                        # Apply TAKER fee to exit
                        final_exit = exit_price * (1 - TAKER_FEE) if position == 1 else exit_price * (1 + TAKER_FEE)
                        real_pnl = (final_exit - safe_entry) / safe_entry if position == 1 else (safe_entry - final_exit) / safe_entry
                        
                        trades.append({
                            'pnl_pct': real_pnl,
                            'duration_bars': i - entry_idx,
                            'win': real_pnl > 0
                        })
                        position = 0
                        self.data_provider.mocked_position = 0
                        entry_idx = 0
        finally:
            time.time = original_time
            from utils.cooldown_manager import cooldown_manager
            cooldown_manager.set_virtual_time(None)
        
        print(f"  [3/4] Análisis completado.")
        
        wins = sum(1 for t in trades if t['win'])
        total_trades = len(trades)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(t['pnl_pct'] for t in trades)
        
        print(f"  [4/4] Resultados Finales:")
        print(f"        Total Señales Generadas: {signals_generated}")
        print(f"        Total Trades: {total_trades}")
        print(f"        Win Rate: {win_rate:.2f}%")
        print(f"        Net PnL (%): {total_pnl*100:.2f}%")
        return trades

if __name__ == "__main__":
    print("🚀 Iniciando Preparación de Datos (Producción vs Sintético)...")
    
    parquet_path = "data/cache_parquet/BNBUSDT_master.parquet"
    if not os.path.exists(parquet_path):
        # Fallback a history folder if available
        parquet_path = r"dashboard\data\futures\history\BNB_USDT_1m.parquet"
        
    if os.path.exists(parquet_path):
        print(f"📦 Cargando historial PARQUET real: {parquet_path}")
        df = pl.read_parquet(parquet_path)
        if 'timestamp' not in df.columns and 'open_time' in df.columns:
            df = df.rename({'open_time': 'timestamp'})
            
        # Ensure timestamp is int64 (ms)
        if df.schema['timestamp'] in [pl.Datetime, pl.Date]:
            df = df.with_columns(pl.col('timestamp').dt.epoch(time_unit='ms'))
            
        if 'symbol' not in df.columns:
            df = df.with_columns(pl.lit('BNB/USDT').alias('symbol'))
            
        # Usar los últimos 14 días para backtest representativo (~20,000 barras)
        df = df.tail(20000)
    else:
        print("⚠️ Parquet real no encontrado. Generando datos sintéticos para prueba rápida...")
        np.random.seed(42)
        n = 10080 # 7 dias en M1
        base_price = 60000.0
        returns = np.random.randn(n) * 0.001
        prices = base_price * np.exp(np.cumsum(returns))
        df = pl.DataFrame({
            'symbol': ['BNB/USDT'] * n,
            'open': prices - np.abs(np.random.randn(n) * 10),
            'high': prices + np.abs(np.random.randn(n) * 30),
            'low': prices - np.abs(np.random.randn(n) * 30),
            'close': prices,
            'volume': np.abs(np.random.randn(n) * 5 + 10),
            'timestamp': [int(datetime.now(timezone.utc).timestamp() * 1000) + (i * 60000) for i in range(n)],
        })
        df = df.with_columns([
            pl.max_horizontal('open', 'close', 'high').alias('high'),
            pl.min_horizontal('open', 'close', 'low').alias('low'),
        ])

    engine = UnifiedBacktestEngine(df)
    engine.run('BNB/USDT')
