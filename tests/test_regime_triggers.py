import unittest
import numpy as np
import pandas as pd
from core.market_regime import MarketRegimeDetector

class TestRegimeTriggers(unittest.TestCase):
    def setUp(self):
        self.detector = MarketRegimeDetector()
        
    def test_learning_factors(self):
        self.assertEqual(self.detector.get_learning_factor('TRENDING_BULL'), 1.0)
        self.assertEqual(self.detector.get_learning_factor('CHOPPY'), 0.0)
        self.assertEqual(self.detector.get_learning_factor('RANGING'), 0.2)
        self.assertEqual(self.detector.get_learning_factor('UNKNOWN_REGIME'), 0.0)

    def test_volatility_shock(self):
        # Create synthetic bars
        # Normal Volatility: High-Low = 1.0
        n = 50
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        df = pd.DataFrame(index=dates)
        df['close'] = np.linspace(100, 105, n)
        df['high'] = df['close'] + 0.5
        df['low'] = df['close'] - 0.5
        df['open'] = df['close']
        
        # ATR approx 1.0
        
        # Scenario 1: No Shock
        # Last bar normal
        bars = df.to_records()
        self.assertFalse(self.detector.is_volatility_shock(bars))
        
        # Scenario 2: Shock
        # Last bar Huge Range (High-Low = 5.0)
        df.iloc[-1, df.columns.get_loc('high')] += 2.5
        df.iloc[-1, df.columns.get_loc('low')] -= 2.5
        
        bars_shock = df.to_records()
        # TR = 5.0 + 1.0 (gap) approx.
        # ATR was 1.0. 
        # 5.0 > 2.5 * 1.0 => True
        self.assertTrue(self.detector.is_volatility_shock(bars_shock))

if __name__ == '__main__':
    unittest.main()
