import unittest
import numpy as np
import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Root path
sys.path.append(os.getcwd())

from strategies.technical import HybridScalpingStrategy
from core.genotype import Genotype
from core.enums import SignalType

class TestFusedStrategy(unittest.TestCase):
    def setUp(self):
        self.data_provider = MagicMock()
        self.events_queue = MagicMock()
        
        # Initialize Genotype for BTC
        self.genotype = Genotype(symbol="BTC/USDT")
        self.genotype.init_brain(25, 4)
        
        self.strategy = HybridScalpingStrategy(
            data_provider=self.data_provider,
            events_queue=self.events_queue,
            genotype=self.genotype
        )
        
    def test_fused_insight_execution(self):
        print("\n🧪 Testing Fused Strategy Execution...")
        
        # Prepare mock data (30 bars)
        n = 50
        data = np.zeros(n, dtype=[
            ('timestamp', 'M8[ms]'),
            ('open', 'f4'), ('high', 'f4'), ('low', 'f4'), 
            ('close', 'f4'), ('volume', 'f4')
        ])
        data['close'] = np.random.randn(n).astype(np.float32) + 100
        data['high'] = data['close'] + 0.1
        data['low'] = data['close'] - 0.1
        data['volume'] = np.random.randn(n).astype(np.float32) * 10 + 1000
        
        # Mock Data Provider
        self.data_provider.get_latest_bars.return_value = data
        self.data_provider.get_active_positions.return_value = {}
        
        # Call Fused Insight
        decision, confidence = self.strategy.get_fused_insight("BTC/USDT", data)
        
        print(f"   - Fused Decision: {decision}")
        print(f"   - Fused Confidence: {confidence:.4f}")
        
        self.assertIn(decision, [None, SignalType.LONG, SignalType.SHORT, "CLOSE"])
        self.assertGreaterEqual(confidence, -100) # Basic range check
        
        print("✅ Fused Strategy Smoke Test OK.")

if __name__ == "__main__":
    unittest.main()
