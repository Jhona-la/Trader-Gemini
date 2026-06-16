import time
import random
import sys
import os

# Add root to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dark_alpha_queue import DarkAlphaQueue

def run_stress_test():
    print("🌑 [DARK ALPHA] Starting Cython Ring Buffer Stress Test...")
    queue = DarkAlphaQueue(halflife=15.0)
    
    # 1. Injection Test (10,000 pushes)
    num_events = 100_000
    print(f"Injecting {num_events:,} synthetic liquidations...")
    
    start_time = time.perf_counter()
    for _ in range(num_events):
        side = random.choice([1, -1])
        size = random.uniform(50_000, 2_000_000)
        queue.push_liquidation(side, size)
    
    end_time = time.perf_counter()
    injection_time = end_time - start_time
    print(f"✅ Injection Time: {injection_time:.6f}s ({num_events/injection_time:,.0f} ops/sec)")
    
    # 2. Calculation Test (Time Decay of 100,000 elements)
    print("Calculating Net Pressure with Exponential Decay over full buffer...")
    start_time = time.perf_counter()
    net_pressure = queue.get_net_pressure()
    end_time = time.perf_counter()
    
    calc_time = end_time - start_time
    print(f"✅ Calculation Time (GIL-Free Math): {calc_time:.6f}s")
    print(f"📊 Net Pressure Result: ${net_pressure:,.2f}")
    print(f"📈 Total Elements Processed: {queue.get_size()}")
    
    if calc_time < 0.05:
        print("🎯 TEST PASSED: Calculation < 50ms overhead. (HFT Standard)")
    else:
        print("⚠️ TEST WARNING: Calculation overhead too high.")

if __name__ == "__main__":
    run_stress_test()
