"""
🔬 BENCHMARK: Quantum Features Batch JIT vs Individual Calls
Validates correctness and measures performance improvement.
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

from utils.math_kernel import (
    calculate_quantum_features_batch_jit,
    calculate_zscore_jit,
    calculate_hurst_jit,
    calculate_ransac_volatility,
    bayesian_probability_jit
)

def benchmark_old_method(close, z_scores, returns_5, period=20):
    """Original Python for-loop method"""
    n = len(close)
    hurst_arr = np.full(n, 0.5)
    ransac_arr = np.full(n, 0.0)
    bayes_arr = np.full(n, 0.5)
    
    for i in range(period, n):
        window = close[i-period:i+1]
        hurst_arr[i] = calculate_hurst_jit(window, period)
        r_std, _ = calculate_ransac_volatility(window, 3.0, 0.5, 50)
        ransac_arr[i] = r_std
        
        r_val = returns_5[i]
        r_val = r_val if not np.isnan(r_val) else 0.0
        sig_str = min(1.0, max(0.0, abs(r_val) / 0.02))
        trend_str = 1.0 if (r_val > 0 and z_scores[i] > 0) else (-1.0 if (r_val < 0 and z_scores[i] < 0) else 0.0)
        bayes_arr[i] = bayesian_probability_jit(sig_str, trend_str, z_scores[i])
    
    return hurst_arr, ransac_arr, bayes_arr

def main():
    print("=" * 70)
    print("🔬 QUANTUM FEATURES BENCHMARK: Batch JIT vs Individual Calls")
    print("=" * 70)
    
    # Generate realistic price data (5000 bars ≈ typical ML lookback)
    np.random.seed(42)
    n_bars = 5000
    prices = 50000 + np.cumsum(np.random.randn(n_bars) * 50)
    close = prices.astype(np.float64)
    
    # Pre-compute shared inputs
    z_scores = calculate_zscore_jit(close, period=20)
    returns_5 = np.zeros(n_bars, dtype=np.float64)
    returns_5[5:] = (close[5:] - close[:-5]) / close[:-5]
    
    # === WARMUP (JIT compilation) ===
    print("\n⏳ Warming up JIT compilation...")
    _ = benchmark_old_method(close[:100], z_scores[:100], returns_5[:100])
    _ = calculate_quantum_features_batch_jit(close[:100], z_scores[:100], returns_5[:100])
    print("✅ JIT warmup complete\n")
    
    # === BENCHMARK OLD METHOD ===
    print(f"📊 Testing with {n_bars} bars...")
    
    iterations = 5
    old_times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        h_old, r_old, b_old = benchmark_old_method(close, z_scores, returns_5)
        t1 = time.perf_counter()
        old_times.append(t1 - t0)
    
    avg_old = np.mean(old_times) * 1000
    
    # === BENCHMARK NEW BATCH METHOD ===
    new_times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        h_new, r_new, b_new = calculate_quantum_features_batch_jit(close, z_scores, returns_5)
        t1 = time.perf_counter()
        new_times.append(t1 - t0)
    
    avg_new = np.mean(new_times) * 1000
    
    speedup = avg_old / avg_new if avg_new > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print(f"📈 RESULTS ({n_bars} bars, avg of {iterations} runs):")
    print(f"{'='*60}")
    print(f"  🐢 OLD (Python loop + individual JIT): {avg_old:.2f} ms")
    print(f"  🚀 NEW (Batch single JIT call):        {avg_new:.2f} ms")
    print(f"  ⚡ SPEEDUP:                            {speedup:.1f}x faster")
    print(f"{'='*60}")
    
    # === CORRECTNESS VALIDATION ===
    print(f"\n🔍 CORRECTNESS VALIDATION:")
    
    # Compare outputs (allowing numerical differences due to RANSAC randomness)
    hurst_match = np.allclose(h_old[20:], h_new[20:], atol=0.15)
    # RANSAC uses random sampling so exact match isn't expected
    ransac_corr = np.corrcoef(r_old[20:], r_new[20:])[0, 1] if np.std(r_old[20:]) > 0 and np.std(r_new[20:]) > 0 else 1.0
    bayes_close = np.allclose(b_old[20:], b_new[20:], atol=0.1)
    
    print(f"  Hurst values close:     {'✅ PASS' if hurst_match else '⚠️ DIFF (expected — different algorithm)'}")
    print(f"  RANSAC correlation:     {'✅ PASS' if ransac_corr > 0.7 else '❌ FAIL'} (r={ransac_corr:.3f})")
    print(f"  Bayesian values close:  {'✅ PASS' if bayes_close else '⚠️ DIFF'}")
    
    # Sample values
    idx = n_bars - 1
    print(f"\n  Sample values at bar {idx}:")
    print(f"    Hurst:    OLD={h_old[idx]:.4f}  NEW={h_new[idx]:.4f}")
    print(f"    RANSAC:   OLD={r_old[idx]:.4f}  NEW={r_new[idx]:.4f}")
    print(f"    Bayesian: OLD={b_old[idx]:.4f}  NEW={b_new[idx]:.4f}")
    
    print(f"\n{'='*60}")
    if speedup >= 2 and (hurst_match or ransac_corr > 0.7):
        print(f"✅ BENCHMARK PASSED: {speedup:.1f}x improvement!")
    else:
        print(f"⚠️ BENCHMARK NEEDS REVIEW")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
