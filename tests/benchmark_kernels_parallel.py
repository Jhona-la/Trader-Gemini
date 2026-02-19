import time
import numpy as np
import sys
import os

# Root path
sys.path.append(os.getcwd())

from core.online_learning_kernels import jit_sgd_single, jit_sgd_parallel

def benchmark_parallel_dispatch():
    print("\n🚀 Benchmarking Multi-Threaded Kernel Dispatch (Phase 61)...")
    
    n_symbols = 100
    n_weights = 100
    input_dim = 100
    
    # 1. Warm-up
    weights = np.random.randn(n_symbols, n_weights).astype(np.float64)
    states = np.random.randn(n_symbols, input_dim).astype(np.float64)
    actions = np.random.randn(n_symbols).astype(np.float64)
    rewards = np.random.randn(n_symbols).astype(np.float64)
    next_states = np.random.randn(n_symbols, input_dim).astype(np.float64)
    
    # Trigger JIT
    jit_sgd_parallel(weights, states, actions, rewards, next_states, 0.01, 1.0, 0.95)
    
    # 2. Serial Execution (Simulated via loop)
    start_serial = time.perf_counter()
    for i in range(n_symbols):
        jit_sgd_single(weights[i], states[i], actions[i], rewards[i], next_states[i], 0.01, 1.0, 0.95)
    end_serial = time.perf_counter()
    serial_time = (end_serial - start_serial) * 1e6 # microseconds
    
    # 3. Parallel Execution
    start_parallel = time.perf_counter()
    jit_sgd_parallel(weights, states, actions, rewards, next_states, 0.01, 1.0, 0.95)
    end_parallel = time.perf_counter()
    parallel_time = (end_parallel - start_parallel) * 1e6 # microseconds

    
    print(f"   - Serial time (Batch size {n_symbols}): {serial_time:.2f} μs")
    print(f"   - Parallel time (Batch size {n_symbols}): {parallel_time:.2f} μs")
    print(f"   - Speedup: {serial_time / parallel_time:.2f}x")
    
    if parallel_time < serial_time:
        print("✅ Parallel dispatch is faster.")
    else:
        print("⚠️ Parallel dispatch overhead exceeds gain for this batch size.")

if __name__ == "__main__":
    benchmark_parallel_dispatch()
