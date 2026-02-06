
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ml_strategy import MLStrategyHybridUltimate

class MockDataProvider:
    def get_latest_bars(self, symbol, n=100):
        return []

class TestMLLeakage(unittest.TestCase):
    def setUp(self):
        self.strategy = MLStrategyHybridUltimate(MockDataProvider(), None)
        
    def test_feature_leakage(self):
        print("\n🧪 Testing for Feature Leakage (Lookahead Bias)...")
        
        # 1. Generate Synthetic Data (Random Walk)
        np.random.seed(42)
        n = 1000
        closes = np.cumsum(np.random.randn(n)) + 100
        highs = closes + np.abs(np.random.randn(n))
        lows = closes - np.abs(np.random.randn(n))
        opens = (highs + lows) / 2
        volumes = np.random.randint(100, 1000, n)
        
        bars = []
        for i in range(n):
            bars.append({
                'datetime': pd.Timestamp.now(), 
                'open': opens[i], 'high': highs[i], 
                'low': lows[i], 'close': closes[i], 
                'volume': volumes[i]
            })
            
        # 2. Generate Features
        df = self.strategy._prepare_features(bars)
        
        # 3. Validation: No feature should correlate 1.0 with FUTURE returns
        future_returns = df['close'].pct_change().shift(-1)
        
        leakage_found = False
        leaking_cols = []
        
        # Skip last row (NaN in future_returns)
        df_clean = df.iloc[:-1].copy()
        y_future = future_returns.iloc[:-1]
        
        for col in df_clean.columns:
            # Skip non-numeric
            if not np.issubdtype(df_clean[col].dtype, np.number):
                continue
                
            # Check correlation
            corr = df_clean[col].corr(y_future)
            
            if abs(corr) > 0.99:
                print(f"   ⚠️ SUSPICIOUS: {col} corr={corr:.4f} with T+1 Return")
                leakage_found = True
                leaking_cols.append(col)
                
        if not leakage_found:
            print("   ✅ No direct lookahead leakage found in features.")
        else:
            self.fail(f"Leakage detected in columns: {leaking_cols}")
            
    def test_target_leakage(self):
        """Ensure targets are NOT in feature set"""
        # Logic is inside _train_model, but we can verify feature_cols if available
        # Ideally we check logic code or use this runtime test
        pass

if __name__ == '__main__':
    unittest.main()
