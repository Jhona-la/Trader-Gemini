import polars as pl
import traceback
import sys

from strategies.components.feature_engineering import FeatureEngineering

df = pl.DataFrame({
    'symbol':['BTC/USDT']*50, 
    'close': range(50), 
    'high': range(50), 
    'low': range(50), 
    'volume': range(50), 
    'open': range(50), 
    'timestamp': range(50)
})
df = df.with_columns([pl.col(c).cast(pl.Float64) for c in ['close', 'high', 'low', 'volume', 'open']])

f = FeatureEngineering()
try:
    f.prepare_features(df)
except Exception as e:
    traceback.print_exc()
