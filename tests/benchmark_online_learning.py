import time
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from core.online_learning import OnlineLearner
from core.online_learning_kernels import jit_sgd_single

def benchmark_single_step():
    print("\n⚡ Benchmarking Single-Step SGD (<500ns Target)...")
    
    # Setup Data (float32 for AVX2)
    input_dim = 25
    output_dim = 4
    n_weights = input_dim * output_dim
    
    weights = np.zeros(n_weights, dtype=np.float32)
    state = np.random.random(input_dim).astype(np.float32)
    next_state = np.random.random(input_dim).astype(np.float32)
    action = 2.0 # Action Index
    reward = 1.0
    
    lr = 0.01
    clip = 0.05
    gamma = 0.99
    
    # Warmup JIT
    jit_sgd_single(weights, state, action, reward, next_state, lr, clip, gamma)
    
    # Validation Run (Check if it runs)
    print("   > JIT Warmup Complete.")
    
    # Benchmarking Loop
    iterations = 1_000_000
    
    start_ns = time.perf_counter_ns()
    
    # We call the kernel directly to measure "Metal-Core" speed
    # Calling via class method adds Python overhead (~200ns)
    # The goal is to prove the KERNEL is fast.
    
    for _ in range(iterations):
        jit_sgd_single(weights, state, action, reward, next_state, lr, clip, gamma)
        
    end_ns = time.perf_counter_ns()
    
    total_ns = end_ns - start_ns
    avg_ns = total_ns / iterations
    
    print(f"   > Iterations: {iterations}")
    print(f"   > Total Time: {total_ns/1e6:.2f} ms")
    print(f"   > Avg Latency: {avg_ns:.2f} ns")
    
    if avg_ns < 500:
        print("✅ SUCCESS: Sub-500ns Latency Achieved!")
    else:
        print(f"⚠️ WARNING: Latency {avg_ns:.2f}ns > 500ns Target.")

if __name__ == "__main__":
    benchmark_single_step()
