import time
import numpy as np

# Python fallback logic
def calculate_python(closes):
    import talib
    rsi = talib.RSI(closes, timeperiod=14)
    eff_window = min(20, len(closes))
    slice_closes = closes[-eff_window:]
    mean_last = np.mean(slice_closes)
    std_last = np.std(slice_closes, ddof=0)
    curr_z = (closes[-1] - mean_last) / std_last if std_last > 0 else 0.0
    returns = np.diff(np.log(closes), prepend=np.log(closes[0]))
    return rsi[-1], curr_z, returns[-1]

try:
    from core.hyper_kernel import HyperKernel
    hk = HyperKernel()
    print("✅ HyperKernel loaded successfully!")
    
    # Generate random prices
    np.random.seed(42)
    test_prices = np.random.uniform(10, 20, 250).astype(np.float64)
    
    print("\n--- Correctness Test ---")
    py_rsi, py_z, py_ret = calculate_python(test_prices)
    hk_rsi, hk_z, hk_ret = hk.batch_update_and_calculate(test_prices)
    
    print(f"Python -> RSI: {py_rsi:.4f}, Z: {py_z:.4f}, Ret: {py_ret:.6f}")
    print(f"Kernel -> RSI: {hk_rsi:.4f}, Z: {hk_z:.4f}, Ret: {hk_ret:.6f}")
    
    print("\n--- Benchmark (10,000 inferences) ---")
    
    # Python benchmark
    start = time.perf_counter_ns()
    for _ in range(10000):
        calculate_python(test_prices)
    py_time = (time.perf_counter_ns() - start) / 1e6
    print(f"Python Time: {py_time:.2f} ms")
    
    # Cython benchmark
    start = time.perf_counter_ns()
    for _ in range(10000):
        # We simulate the streaming per-tick by doing single updates or batch
        hk.batch_update_and_calculate(test_prices)
    cy_time = (time.perf_counter_ns() - start) / 1e6
    print(f"Cython Time: {cy_time:.2f} ms")
    
    print(f"🚀 Speedup: {py_time/cy_time:.1f}x")
    
except ImportError as e:
    print(f"❌ Failed to load HyperKernel: {e}")
except Exception as e:
    print(f"❌ Error during execution: {e}")
