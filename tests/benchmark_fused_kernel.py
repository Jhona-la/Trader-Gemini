import time
import numpy as np
import sys
import os

# Root path
sys.path.append(os.getcwd())

from core.fused_strategy_kernel import fused_compute_step

def benchmark_fused_kernel():
    print("\n🚀 Benchmarking Kernel Fusion (Phase 65)...")
    
    n_bars = 1000
    closes = np.random.randn(n_bars).astype(np.float32) + 100
    volumes = np.random.randn(n_bars).astype(np.float32) * 100 + 1000
    
    portfolio_state = np.array([1.0, 0.05, 0.2], dtype=np.float32)
    gene_params = np.array([0.02, 0.015], dtype=np.float32)
    brain_weights = np.random.randn(100).astype(np.float32)
    
    # Warm-up
    fused_compute_step(closes, volumes, portfolio_state, gene_params, brain_weights)
    
    # 1. Measure Single Step Latency
    n_iterations = 10000
    start = time.perf_counter()
    for _ in range(n_iterations):
        res = fused_compute_step(closes, volumes, portfolio_state, gene_params, brain_weights)
    end = time.perf_counter()
    
    avg_lat = (end - start) * 1e6 / n_iterations # Microseconds
    print(f"   - Fused Strategy Step Latency: {avg_lat:.4f} μs")
    
    if avg_lat < 5.0:
        print(f"✅ Kernel Fusion Success: Latency is extremely low ({avg_lat:.2f} μs)")
    else:
        print(f"ℹ️ Latency: {avg_lat:.2f} μs")

if __name__ == "__main__":
    benchmark_fused_kernel()
