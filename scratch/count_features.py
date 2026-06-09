import numpy as np
import pandas as pd
from strategies.components.feature_engineering import FeatureEngineering

# Create dummy bars
n = 300
bars = {
    'open': np.linspace(100, 110, n),
    'high': np.linspace(101, 111, n),
    'low': np.linspace(99, 109, n),
    'close': np.linspace(100.5, 110.5, n),
    'volume': np.linspace(1000, 1500, n),
    'datetime': pd.date_range(start='2026-05-19', periods=n, freq='1min')
}

fe = FeatureEngineering()
df = fe.prepare_features(bars, symbol='BTC/USDT')

print("=== Output columns count ===")
print("Number of columns:", len(df.columns))
print("Columns list:", sorted(list(df.columns)))
