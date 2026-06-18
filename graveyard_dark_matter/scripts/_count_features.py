import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from strategies.components.feature_engineering import FeatureEngineering

n = 200
np.random.seed(42)
prices = np.random.randn(n).cumsum() + 100
df = pl.DataFrame({
    'close': prices,
    'open': prices + np.random.randn(n)*0.1,
    'high': prices + abs(np.random.randn(n)*0.5),
    'low': prices - abs(np.random.randn(n)*0.5),
    'volume': np.abs(np.random.randn(n)) * 1000,
    'timestamp': list(range(n))
})

fe = FeatureEngineering()
out = fe.prepare_features(df, symbol='BTC/USDT', horizon='SCALPING')
print(f"OUTPUT COLUMNS: {len(out.columns)}")
print("COLUMNS:", list(out.columns))
