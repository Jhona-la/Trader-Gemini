"""
Test de verificación post-auditoría: Features Resucitadas
Verifica que las features antes muertas ahora producen valores no-cero.
"""
import sys
sys.path.insert(0, '.')

import polars as pl
import numpy as np
from strategies.components.feature_engineering import FeatureEngineering

# Crear datos sintéticos realistas (100 velas de 1m para BTCUSDT)
np.random.seed(42)
n = 200
base_price = 105000.0
prices = base_price + np.cumsum(np.random.randn(n) * 50)

df = pl.DataFrame({
    'symbol': ['BTCUSDT'] * n,
    'open': prices - np.random.rand(n) * 20,
    'high': prices + np.abs(np.random.randn(n) * 30),
    'low': prices - np.abs(np.random.randn(n) * 30),
    'close': prices,
    'volume': np.abs(np.random.randn(n) * 100 + 50),
    'timestamp': list(range(n)),
})

# Asegurar que high >= max(open,close) y low <= min(open,close)
df = df.with_columns([
    pl.max_horizontal('open', 'close', 'high').alias('high'),
    pl.min_horizontal('open', 'close', 'low').alias('low'),
])

fe = FeatureEngineering()
result = fe.prepare_features(df, horizon='SCALPING')

print("=" * 70)
print("🔬 TEST DE FEATURES RESUCITADAS")
print("=" * 70)

# Features que ANTES estaban muertas y ahora deben tener varianza
resurrected = [
    'tick_direction', 'net_pressure', 'micro_velocity_3', 'volume_accel',
    'vbi', 'vbi_avg', 'trend_power', 'trend_alignment', 'range_extreme',
    'panic_index', 'scalp_rsi_divergence', 'is_swing_horizon',
    'swing_momentum_ratio', 'swing_ema50_slope', 'spread_squeeze',
    'volatility_regime'
]

alive_count = 0
dead_count = 0

for feat in resurrected:
    if feat not in result.columns:
        print(f"  ❌ {feat:30s} — COLUMNA AUSENTE")
        dead_count += 1
        continue
    
    col = result[feat]
    vals = col.to_numpy()
    
    # Check if constant zero
    non_zero = np.count_nonzero(~np.isnan(vals) & (vals != 0.0))
    variance = np.nanvar(vals)
    mean = np.nanmean(vals)
    
    if non_zero == 0:
        print(f"  ❌ {feat:30s} — MUERTA (todos 0.0)")
        dead_count += 1
    else:
        print(f"  ✅ {feat:30s} — VIVA | non-zero={non_zero:4d}/{n} | mean={mean:+.6f} | var={variance:.6f} | dtype={col.dtype}")
        alive_count += 1

# Features que deben seguir como 0.0 (requieren API)
api_features = ['funding_rate', 'oi', 'l2_ofi', 'dark_alpha_pressure', 'onchain_whale_flow']
print(f"\n  Features API (documentadas como 0.0):")
for feat in api_features:
    if feat in result.columns:
        vals = result[feat].to_numpy()
        is_zero = np.all(vals == 0.0)
        status = "0.0 (esperado)" if is_zero else f"NON-ZERO: {np.nanmean(vals):.4f}"
        print(f"    📡 {feat:30s} — {status}")

# Check dtype preservation
print(f"\n  📊 DTYPE CHECK (float64 preservado?):")
float64_cols = [c for c, d in zip(result.columns, result.dtypes) if d == pl.Float64]
float32_cols = [c for c, d in zip(result.columns, result.dtypes) if d == pl.Float32]
print(f"    Float64 columns: {len(float64_cols)}")
print(f"    Float32 columns: {len(float32_cols)}")
if float32_cols:
    print(f"    ⚠️ Float32 columns found: {float32_cols[:10]}...")

print(f"\n{'=' * 70}")
print(f"  RESUCITADAS: {alive_count}/{len(resurrected)}")
print(f"  AÚN MUERTAS: {dead_count}/{len(resurrected)}")
print(f"  TOTAL COLUMNAS: {len(result.columns)}")
print(f"  NaN count: {result.null_count().sum_horizontal()[0]}")
print(f"{'=' * 70}")
