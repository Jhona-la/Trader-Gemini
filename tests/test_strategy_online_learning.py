import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from strategies.technical import HybridScalpingStrategy
from core.genotype import Genotype
from core.events import MarketEvent

class MockDataProvider:
    def __init__(self):
        self.symbol_list = ['BTC/USDT']
        self.positions = {}
    
    def get_active_positions(self):
        return self.positions
        
    def get_latest_bars(self, symbol, n=300, timeframe='5m'):
        # Generate synthetic data (Structured Array)
        dates = pd.date_range(start='2024-01-01', periods=n, freq=timeframe)
        dtype = [('timestamp', 'datetime64[us]'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')]
        data = np.zeros(n, dtype=dtype)
        
        
        # Simple trend with NOISE
        prices = np.linspace(100, 110, n)
        noise = np.random.normal(0, 0.5, n)
        prices = prices + noise
        
        data['timestamp'] = dates
        data['open'] = prices
        data['high'] = prices + 1
        data['low'] = prices - 1
        data['close'] = prices
        data['volume'] = 1000.0 + np.random.normal(0, 100, n)
        
        return data

class MockQueue:
    def put(self, item):
        pass

class TestStrategyOnlineLearning(unittest.TestCase):
    def test_learning_loop(self):
        # 1. Setup
        dp = MockDataProvider()
        queue = MockQueue()
        
        # Genes with Brain
        genotype = Genotype('BTC/USDT')
        genotype.init_brain(25, 4) # Input 25, Output 4
        
        # Add required genes for Strategy
        genotype.genes.update({
            'adx_threshold': 20,
            'strength_threshold': 0.6,
            'tp_pct': 0.015,
            'sl_pct': 0.02
        })
        
        orig_weights = np.array(genotype.genes['brain_weights']).copy()
        
        strategy = HybridScalpingStrategy(dp, queue, genotype=genotype)
        strategy.learner.learning_rate = 0.5 # Hyper-aggressive for test
        
        # 2. Run Tick 1 (Initial Prediction, No Learning yet)
        event1 = MarketEvent(symbol='BTC/USDT', close_price=100.0, timestamp=datetime.now(timezone.utc))
        strategy.generate_signals(event1)
        
        # Verify Memory
        self.assertIn('BTC/USDT', strategy.brain_memory)
        
        # 3. Run Tick 2 (Should Trigger Learning from Tick 1)
        # Price moved up -> Should learn positive reinforcement (?)
        event2 = MarketEvent(symbol='BTC/USDT', close_price=101.0, timestamp=datetime.now(timezone.utc))
        strategy.generate_signals(event2)
        
        # 4. Verify Weights Changed
        new_weights = np.array(genotype.genes['brain_weights'])
        
        # There should be SOME difference
        diff = np.sum(np.abs(new_weights - orig_weights))
        print(f"Weight Delta: {diff}")
        self.assertTrue(diff > 0.0, "Brain weights should have updated!")

if __name__ == '__main__':
    unittest.main()
