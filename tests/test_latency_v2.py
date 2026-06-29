import time
import numpy as np
import asyncio
from core.cython_bridge.nano_ffi import NanoFFIBridge

async def run_latency_test():
    print("🚀 [TEST] Initiating V2 Pure Metal Latency Test...")
    try:
        bridge = NanoFFIBridge()
    except Exception as e:
        print(f"❌ [ERROR] Bridge failed to initialize: {e}")
        return

    # Simulate 1024-bar OHLCV tensor
    prices = np.random.randn(1024).astype(np.float32)
    volumes = np.random.randn(1024).astype(np.float32)

    num_iterations = 100_000
    print(f"⚛️ Running {num_iterations} iterations...")
    
    start_time = time.perf_counter_ns()
    
    for _ in range(num_iterations):
        _ = bridge.invoke_oracle(
            prices, volumes,
            mempool_panic=0.1,
            net_liq=-0.2,
            timestamp=time.time_ns()
        )
        
    end_time = time.perf_counter_ns()
    
    total_time_ms = (end_time - start_time) / 1_000_000
    latency_per_tick_ns = (end_time - start_time) / num_iterations
    
    print(f"✅ [SUCCESS] V2 Engine Stress Test Completed.")
    print(f"📊 Total Time for {num_iterations} ticks: {total_time_ms:.2f} ms")
    print(f"⚡ Latency per tick: {latency_per_tick_ns:.2f} ns")
    
    if latency_per_tick_ns > 10_000_000: # 10 ms
        print("❌ [FAIL] Latency exceeds 10ms threshold.")
        import os
        os._exit(137)
    else:
        print("🏆 [PASS] Latency is sub-10ms as required by the Dogma.")

if __name__ == "__main__":
    asyncio.run(run_latency_test())
