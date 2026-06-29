import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from strategies.technical import HybridScalpingStrategy
from core.genotype import Genotype
from core.events import MarketEvent

class MockDataProvider:
    def __init__(self):
        self.symbol_list = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.positions = {}
    
    def get_active_positions(self):
        return self.positions
    
    def get_latest_bars(self, symbol, n=300, timeframe='5m'):
        # Generate synthetic data
        dates = pd.date_range(start='2024-01-01', periods=n, freq=timeframe)
        dtype = [('timestamp', 'datetime64[us]'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')]
        data = np.zeros(n, dtype=dtype)
        
        # Different price levels for symbols
        base_price = 100.0
        if symbol == 'ETH/USDT': base_price = 2000.0
        if symbol == 'SOL/USDT': base_price = 50.0
        
        # Trend with noise
        prices = np.linspace(base_price, base_price * 1.05, n) + np.random.normal(0, base_price*0.005, n)
        
        data['timestamp'] = dates
        data['open'] = prices
        data['high'] = prices + (base_price * 0.01)
        data['low'] = prices - (base_price * 0.01)
        data['close'] = prices
        data['volume'] = 1000.0
        
        return data

class MockQueue:
    def put(self, item): pass

class TestSmokeMultiverse(unittest.TestCase):
    def test_polars_engine(self):
        """
        Phase 16: Verify Polars Integration (Rust Engine).
        """
        print("\n🐼 Testing Polars Engine...")
        
        # Initialize BinanceData with Mock Queue
        from data.binance_loader import BinanceData
        from core.events import MarketEvent
        from utils.hft_buffer import NumbaStructuredRingBuffer
        import asyncio
        import pytest
        import polars as pl
        from unittest.mock import MagicMock
        
        # Mocking Connection
        original_init = BinanceData._init_sync_client
        original_fetch = BinanceData.fetch_initial_history
        original_fetch_1h = BinanceData.fetch_initial_history_1h
        original_fetch_5m = BinanceData.fetch_initial_history_5m
        original_fetch_15m = BinanceData.fetch_initial_history_15m
        
        BinanceData._init_sync_client = lambda self: None
        BinanceData.fetch_initial_history = lambda self: None
        BinanceData.fetch_initial_history_1h = lambda self: None
        BinanceData.fetch_initial_history_5m = lambda self: None
        BinanceData.fetch_initial_history_15m = lambda self: None
        
        try:
            # Mock Queue
            queue = asyncio.Queue()
            loader = BinanceData(queue, ['BTC/USDT'])
            
            # Manual Init of Attributes skipped by mocked _init_sync_client
            loader.buffers_1m = {}
            loader.buffers_5m = {}
            loader.buffers_15m = {}
            loader.buffers_1h = {}
            
            # Manual Buffer Init for Symbol
            from utils.hft_buffer import NumbaStructuredRingBuffer
            if 'BTC/USDT' not in loader.buffers_1m:
                loader.buffers_1m['BTC/USDT'] = NumbaStructuredRingBuffer(100)
            
            # Test 1: Empty Buffer
            df_empty = loader.get_history_polars('BTC/USDT')
            self.assertTrue(df_empty.is_empty())
            
            # Test 2: Populated Buffer
            buf = loader.buffers_1m['BTC/USDT']
            
            # Push dummy data
            for i in range(10):
                buf.push(1000+i, 100.0, 105.0, 95.0, 102.0, 1.0)
                
            # Retrieve as Polars
            df = loader.get_history_polars('BTC/USDT', n=5)
            
            print(f"   > Schema: {df.schema}")
            print(f"   > Auto-Cast Int64: {df['timestamp'].dtype}")
            
            self.assertEqual(len(df), 5)
            self.assertEqual(df['close'][-1], 102.0)
            # Polars Int64 check
            self.assertTrue(df['timestamp'].dtype == pl.Int64)
            print("OK > Polars DataFrame Constructed via Zero-Copy.")
            
        finally:
            # Restore
            BinanceData._init_sync_client = original_init
            BinanceData.fetch_initial_history = original_fetch
            BinanceData.fetch_initial_history_1h = original_fetch_1h
            BinanceData.fetch_initial_history_5m = original_fetch_5m
            BinanceData.fetch_initial_history_15m = original_fetch_15m


    def test_multiverse_learning(self):
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        dp = MockDataProvider()
        queue = MockQueue()
        
        # SINGLE STRATEGY INSTANCE (The "Multiverse" Container)
        strategy = HybridScalpingStrategy(dp, queue)
        strategy.learner.learning_rate = 0.5 
        
        print("\n🌌 Initializing Multiverse (Singleton Strategy)...")
        
        # 1. Spawn Brains via get_symbol_params
        orig_weights = {}
        for symbol in symbols:
            # This triggers auto-spawn
            strategy.get_symbol_params(symbol)
            # Inject required genes for test
            strategy.genotypes[symbol].genes.update({
                'adx_threshold': 20, 'strength_threshold': 0.6, 'tp_pct': 0.015, 'sl_pct': 0.02
            })
            
            orig_weights[symbol] = np.array(strategy.genotypes[symbol].genes['brain_weights']).copy()
            print(f"🚀 Spawning {symbol} (Brain Size: {len(orig_weights[symbol])})")
            
        # 2. Simulate Market Loop
        print("⏳ Running Simulation Loop (5 Ticks)...")
        from datetime import timedelta
        base_time = datetime.now(timezone.utc)
        
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            # print(f"   Tick {i+1}")
            for symbol in symbols:
                # Randomize price slightly for current tick
                base = 100 if 'BTC' in symbol else (2000 if 'ETH' in symbol else 50)
                price = base * (1 + (i*0.01)) # Trend up
                
                event = MarketEvent(symbol=symbol, close_price=price, timestamp=timestamp)
                strategy.generate_signals(event)
                
        # 3. Verify Learning in ALL Strategies
        print("🧠 Verifying Neural Plasticity...")
        for symbol in symbols:
            new_weights = np.array(strategy.genotypes[symbol].genes['brain_weights'])
            diff = np.sum(np.abs(new_weights - orig_weights[symbol]))
            print(f"   > {symbol}: Weight Delta = {diff:.6f}")
            
            self.assertTrue(diff > 0.0, f"{symbol} Brain failed to learn!")
            
        # 4. Verify Persistence
        print("💾 Verifying Shutdown Persistence...")
        strategy.stop()
        for symbol in symbols:
            import os
            self.assertTrue(os.path.exists(f"data/genotypes/{symbol.replace('/','')}_gene.json"))
            
        print("✅ MULTIVERSE SMOKE TEST PASSED")

if __name__ == '__main__':
    unittest.main()
