import time
import numpy as np
import sys
import os

# Root path
sys.path.append(os.getcwd())

from utils.evolution_kernels import jit_crossover_mutation_batch

def benchmark_evolution_parallel():
    print("\n🚀 Benchmarking Evolution Parallel Dispatch (Phase 61)...")
    
    n_genes = 100
    n_offspring = 5000
    
    parent1 = np.random.randn(n_genes).astype(np.float32)
    parent2 = np.random.randn(n_genes).astype(np.float32)
    
    # 1. Warm-up
    jit_crossover_mutation_batch(parent1, parent2, 0.5, 0.1, 0.05, 100)
    
    # 2. Parallel Execution (prange version)
    start_parallel = time.perf_counter()
    jit_crossover_mutation_batch(parent1, parent2, 0.5, 0.1, 0.05, n_offspring)
    end_parallel = time.perf_counter()
    parallel_time = (end_parallel - start_parallel) * 1e3 # ms
    
    print(f"   - Parallel time (Offspring {n_offspring}): {parallel_time:.2f} ms")
    print(f"✅ Evolution parallel dispatch OK.")

if __name__ == "__main__":
    benchmark_evolution_parallel()
