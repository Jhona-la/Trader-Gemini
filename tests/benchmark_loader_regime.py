import time
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

# Target functions
from utils.math_kernel import compute_fuzzy_regime_scores_jit, pearson_correlation_jit

print("=========================================================")
print("🧪 NANO-LATENCY BENCHMARK: REGIME & WEBSOCKET LOADER")
print("=========================================================")

# Warmup JIT
_ = compute_fuzzy_regime_scores_jit(25.0, 0.6, 1.0, True)
_ = pearson_correlation_jit(np.array([1.0, 2.0], dtype=np.float64), np.array([2.0, 3.0], dtype=np.float64))


# 1. MARKET REGIME: PURO PYTHON DICTS VS NUMBA JIT
def old_fuzzy_logic(adx, hurst, tm, is_bullish):
    adx_base = 20.0 * tm
    adx_range = 10.0 * tm
    mr_adx_base = 22.0 * tm
    
    p_trend_adx = max(0.0, min(1.0, (adx - adx_base) / adx_range))
    p_trend_hurst = max(0.0, min(1.0, (hurst - 0.5) / 0.15))
    score_trending = (p_trend_adx * 0.6) + (p_trend_hurst * 0.4)
    
    p_mr_hurst = max(0.0, min(1.0, (0.45 - hurst) / 0.1))
    p_mr_adx = max(0.0, min(1.0, (mr_adx_base - adx) / 7.0))
    score_mean_reverting = (p_mr_hurst * 0.7) + (p_mr_adx * 0.3)
    
    p_range_adx = max(0.0, min(1.0, (mr_adx_base - adx) / 7.0))
    dist_to_neutral = abs(hurst - 0.5)
    p_range_hurst = max(0.0, min(1.0, (0.1 - dist_to_neutral) / 0.1))
    score_ranging = (p_range_adx * 0.5) + (p_range_hurst * 0.5)
    
    score_choppy = max(0.0, 1.0 - max(score_trending, score_mean_reverting, score_ranging))
    
    scores = {
        'TRENDING_BULL' if is_bullish else 'TRENDING_BEAR': score_trending,
        'MEAN_REVERTING': score_mean_reverting,
        'RANGING': score_ranging,
        'CHOPPY': score_choppy
    }
    best_regime = max(scores, key=scores.get)
    if scores[best_regime] < 0.35: return 'CHOPPY'
    return best_regime

n_iter = 10000
t0 = time.perf_counter()
for _ in range(n_iter):
    res_old = old_fuzzy_logic(28.5, 0.66, 1.0, True)
t1 = time.perf_counter()
old_regime_time = (t1 - t0) * 1e6 / n_iter

regime_map = ['TRENDING_BEAR', 'TRENDING_BULL', 'MEAN_REVERTING', 'RANGING', 'CHOPPY']
t0 = time.perf_counter()
for _ in range(n_iter):
    idx, score = compute_fuzzy_regime_scores_jit(28.5, 0.66, 1.0, True)
    res_new = regime_map[idx]
t1 = time.perf_counter()
new_regime_time = (t1 - t0) * 1e6 / n_iter

print(f"\n[1] MARKET REGIME FUZZY LOGIC (Python Dicts vs Numba Float64)")
print(f"  🐢 OLD (Python Dict): {old_regime_time:8.2f} μs")
print(f"  🚀 NEW (Numba JIT):   {new_regime_time:8.2f} μs")
print(f"  ⚡ SPEEDUP:            {old_regime_time/new_regime_time:8.1f}x FASTER")
print(f"  🔍 MATCH:            {'✅ PASS' if res_old == res_new else '❌ FAIL'} ({res_old} vs {res_new})")


# 2. WEBSOCKET LOADER: PANDAS TO_DATETIME VS INTEGER ARITHMETIC
t_str = "1711958400000" # Some epoch ms

t0 = time.perf_counter()
for _ in range(n_iter):
    ts_pd = pd.to_datetime(int(t_str), unit='ms')
t1 = time.perf_counter()
old_pd_time = (t1 - t0) * 1e6 / n_iter

t0 = time.perf_counter()
for _ in range(n_iter):
    ts_int = int(t_str)
t1 = time.perf_counter()
new_int_time = (t1 - t0) * 1e6 / n_iter

print(f"\n[2] WEBSOCKET PARSER DATE ALLOCATION (Pandas Objects vs Int)")
print(f"  🐢 OLD (pd.to_datetime): {old_pd_time:8.2f} μs")
print(f"  🚀 NEW (pure int()):      {new_int_time:8.4f} μs")
print(f"  ⚡ SPEEDUP:               {old_pd_time/new_int_time:8.1f}x FASTER")


# 3. WEBSOCKET LEAD-LAG: NP.CORRCOEF VS PEARSON JIT
x_arr = np.random.randn(60).astype(np.float64)
y_arr = np.random.randn(60).astype(np.float64)

t0 = time.perf_counter()
for _ in range(n_iter):
    corr_old = np.corrcoef(x_arr, y_arr)[0, 1]
t1 = time.perf_counter()
old_corr_time = (t1 - t0) * 1e6 / n_iter

t0 = time.perf_counter()
for _ in range(n_iter):
    corr_new = pearson_correlation_jit(x_arr, y_arr)
t1 = time.perf_counter()
new_corr_time = (t1 - t0) * 1e6 / n_iter

print(f"\n[3] CROSS CORRELATION 60-BAR (np.corrcoef vs Pearson JIT)")
print(f"  🐢 OLD (np.corrcoef): {old_corr_time:8.2f} μs")
print(f"  🚀 NEW (Pearson JIT): {new_corr_time:8.2f} μs")
print(f"  ⚡ SPEEDUP:            {old_corr_time/new_corr_time:8.1f}x FASTER")
print(f"  🔍 MATCH:            {'✅ PASS' if abs(corr_old - corr_new) < 1e-6 else '❌ FAIL'}")

print("=========================================================")
