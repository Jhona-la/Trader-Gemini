import sys
import os
import time
import numpy as np

# Add the directory containing the nano_core.pyd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/rust_core')))
import nano_core

def test_rust_engine():
    engine = nano_core.StatefulEngine(period_ema=20, period_rsi=14, capacity=1000)
    
    prices = np.random.uniform(90000, 95000, 100).astype(np.float64)
    
    t0 = time.perf_counter()
    engine.seed_history(prices)
    t1 = time.perf_counter()
    print(f"Seed history: {(t1-t0)*1000:.4f} ms")
    
    # Simulate a Quantum Arena
    features = np.zeros((1000, 200), dtype=np.float32)
    version = np.zeros(8, dtype=np.int64)
    reader_head = np.array([-1], dtype=np.int64)
    feature_vec = np.random.uniform(0, 1, 200).astype(np.float32)
    
    t0 = time.perf_counter()
    for i in range(100000):
        engine.update_and_inject(
            95000.0 + i, 
            version, 
            reader_head, 
            features, 
            feature_vec
        )
    t1 = time.perf_counter()
    
    ops = 100000
    ns_per_op = (t1 - t0) * 1e9 / ops
    
    print(f"100k injections took: {(t1-t0)*1000:.4f} ms")
    print(f"Latency per tick: {ns_per_op:.2f} nanoseconds")
    print(f"Version counter: {version[0]}")
    print(f"Current head: {engine.get_head()}")

if __name__ == "__main__":
    test_rust_engine()
