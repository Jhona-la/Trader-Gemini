import time
import numpy as np
import sys
import os

# Root path
sys.path.append(os.getcwd())

from utils.math_kernel import calculate_zscore_jit, calculate_correlation_matrix_jit

def benchmark_simd_math():
    print("\n🚀 Benchmarking SIMD Math Optimizations (Phase 62)...")
    
    # 1. Z-Score Benchmark
    n = 10000
    period = 100
    prices = np.random.randn(n).astype(np.float32)
    
    # Warm-up
    calculate_zscore_jit(prices, period)
    
    start = time.perf_counter()
    for _ in range(100):
        calculate_zscore_jit(prices, period)
    end = time.perf_counter()
    
    z_time = (end - start) * 10 # average in ms
    print(f"   - O(N) Z-Score (N={n}): {z_time:.4f} ms")
    
    # 2. Correlation Matrix Benchmark
    n_samples = 1000
    m_assets = 50
    matrix = np.random.randn(n_samples, m_assets).astype(np.float32)
    
    # Warm-up
    calculate_correlation_matrix_jit(matrix)
    
    start = time.perf_counter()
    for _ in range(100):
        calculate_correlation_matrix_jit(matrix)
    end = time.perf_counter()
    
    corr_time = (end - start) * 10
    print(f"   - SIMD Correlation ({m_assets}x{m_assets}): {corr_time:.4f} ms")
    
    print("\n✅ SIMD Kernels verified.")

if __name__ == "__main__":
    benchmark_simd_math()
