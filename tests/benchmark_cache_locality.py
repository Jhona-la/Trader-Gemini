import time
import numpy as np
import sys
import os

# Root path
sys.path.append(os.getcwd())

from core.online_learning import OnlineLearner

def benchmark_cache_locality():
    print("\n🚀 Benchmarking Cache-Aware Memory Layout (Phase 63)...")
    
    n_symbols = 256 # Scale up to stress cache
    n_weights = 100
    input_dim = 100
    
    learner = OnlineLearner()
    
    # 1. Non-contiguous Case (Fragmented memory)
    # We force non-contiguity by slicing a larger array with steps
    weights_large = np.random.randn(n_symbols * 2, n_weights).astype(np.float32)
    weights_batch = weights_large[::2, :] # Strided, non-contiguous
    
    states_large = np.random.randn(n_symbols * 2, input_dim).astype(np.float32)
    states_batch = states_large[::2, :]
    
    actions = np.zeros(n_symbols).astype(np.float32)
    rewards = np.zeros(n_symbols).astype(np.float32)
    next_states_batch = states_large[::2, :]
    
    # Warm-up (will trigger np.ascontiguousarray internal check)
    learner.learn_batch(weights_batch, states_batch, actions, rewards, next_states_batch)
    
    start_non = time.perf_counter()
    for _ in range(50):
        learner.learn_batch(weights_batch, states_batch, actions, rewards, next_states_batch)
    end_non = time.perf_counter()
    non_cont_time = (end_non - start_non) * 1000 # ms
    
    # 2. Contiguous Case (Optimized locality)
    weights_cont = np.ascontiguousarray(weights_batch)
    states_cont = np.ascontiguousarray(states_batch)
    ns_cont = np.ascontiguousarray(next_states_batch)
    
    start_cont = time.perf_counter()
    for _ in range(50):
        learner.learn_batch(weights_cont, states_cont, actions, rewards, ns_cont)
    end_cont = time.perf_counter()
    cont_time = (end_cont - start_cont) * 1000 # ms
    
    print(f"   - Non-contiguous Throughput Time: {non_cont_time:.2f} ms")
    print(f"   - Contiguous (Cache-Aware) Time: {cont_time:.2f} ms")
    
    if cont_time < non_cont_time:
        improvement = ((non_cont_time - cont_time) / non_cont_time) * 100
        print(f"✅ Cache locality improvement: {improvement:.2f}%")
    else:
        print("ℹ️ Cache difference negligible for this batch size.")

if __name__ == "__main__":
    benchmark_cache_locality()
