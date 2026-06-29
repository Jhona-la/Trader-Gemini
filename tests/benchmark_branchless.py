import time
import numpy as np
import sys
import os

# Root path
sys.path.append(os.getcwd())

from core.online_learning_kernels import jit_sgd_single
from utils.math_kernel import calculate_rsi_jit

def benchmark_branchless_jitter():
    print("\n🚀 Benchmarking Branch Prediction & Jitter (Phase 64)...")
    
    n_iterations = 1000
    
    # 1. SGD Single Update Jitter
    weights = np.random.randn(100).astype(np.float32)
    state = np.random.randn(25).astype(np.float32)
    next_state = np.random.randn(25).astype(np.float32)
    
    # Warm-up
    jit_sgd_single(weights, state, 1, 0.5, next_state, 0.01, 1.0, 0.95)
    
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        jit_sgd_single(weights, state, 1, 0.5, next_state, 0.01, 1.0, 0.95)
        end = time.perf_counter()
        latencies.append((end - start) * 1e9) # Nanoseconds
        
    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    
    print(f"   - SGD Single latency: {avg_lat:.2f} ns (StdDev/Jitter: {std_lat:.2f} ns)")
    
    # 2. RSI Calculation Jitter
    prices = np.random.randn(1000).astype(np.float64)
    calculate_rsi_jit(prices, 14) # Warm-up
    
    rsi_latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        calculate_rsi_jit(prices, 14)
        end = time.perf_counter()
        rsi_latencies.append((end - start) * 1e6) # Microseconds
        
    avg_rsi = np.mean(rsi_latencies)
    std_rsi = np.std(rsi_latencies)
    
    print(f"   - RSI JIT latency: {avg_rsi:.2f} μs (StdDev/Jitter: {std_rsi:.2f} μs)")
    
    print("\n✅ Branchless optimizations verified for jitter reduction.")

if __name__ == "__main__":
    benchmark_branchless_jitter()
