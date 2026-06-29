import polars as pl
import numpy as np

from strategies.indicators.momentum import MomentumIndicators
from strategies.indicators.trend import TrendIndicators
from strategies.indicators.volatility import VolatilityIndicators
from strategies.indicators.volume import VolumeIndicators
from strategies.indicators.structure import StructureIndicators

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

close = df['close'].to_numpy()
high = df['high'].to_numpy()
low = df['low'].to_numpy()
volume = df['volume'].to_numpy()
n_len = 50

print("Testing Momentum")
try:
    m1 = MomentumIndicators.calculate_all(df, close, high, low, volume, n_len)
    df.with_columns([v.alias(k) for k,v in m1.items() if isinstance(v, pl.Expr)])
except Exception as e:
    print("FAILED MOMENTUM:", e)

print("Testing Trend")
try:
    m2 = TrendIndicators.calculate_all(df, close, high, low, n_len)
    df.with_columns([v.alias(k) for k,v in m2.items() if isinstance(v, pl.Expr)])
except Exception as e:
    print("FAILED TREND:", e)

print("Testing Volatility")
try:
    m3 = VolatilityIndicators.calculate_all(df, close, high, low, n_len)
    df.with_columns([v.alias(k) for k,v in m3.items() if isinstance(v, pl.Expr)])
except Exception as e:
    print("FAILED VOLATILITY:", e)

print("Testing Volume")
try:
    m4 = VolumeIndicators.calculate_all(df, close, high, low, volume, n_len)
    df.with_columns([v.alias(k) for k,v in m4.items() if isinstance(v, pl.Expr)])
except Exception as e:
    print("FAILED VOLUME:", e)

print("Testing Structure")
try:
    m5 = StructureIndicators.calculate_all(df, close, high, low, n_len)
    df.with_columns([pl.Series(k, v) for k,v in m5.items()])
except Exception as e:
    print("FAILED STRUCTURE:", e)

print("DONE")
