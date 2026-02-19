import time
import numpy as np
import sys
import os

# Root path
sys.path.append(os.getcwd())

from risk.risk_manager import RiskManager
from core.events import SignalEvent
from core.enums import SignalType

def benchmark_risk_latency():
    print("\n🚀 Benchmarking RiskManager Latency (Phase 56)...")
    
    rm = RiskManager()
    # Mock portfolio to avoid re-entering portfolio logic
    rm.portfolio = None 
    
    from datetime import datetime, timezone
    signal = SignalEvent(
        strategy_id="BENCHMARK",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.8,
        atr=0.5
    )
    
    # 1. Warm-up & Cache Init
    rm.generate_order(signal, 100.0)
    
    # 2. Measure generate_order Latency (The Critical Path)
    n_iterations = 5000
    latencies = []
    
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        rm.generate_order(signal, 100.0)
        end = time.perf_counter_ns()
        latencies.append(end - start)
        
    avg_ns = np.mean(latencies)
    std_ns = np.std(latencies)
    
    print(f"   - generate_order Average: {avg_ns/1000:.2f} μs")
    print(f"   - generate_order Jitter (StdDev): {std_ns/1000:.2f} μs")
    
    if avg_ns < 20000: # Target < 20μs for risk logic
        print(f"✅ RiskManager Optimization Success: {avg_ns/1000:.2f} μs")
    else:
        print(f"ℹ️ Current Latency: {avg_ns/1000:.2f} μs")

if __name__ == "__main__":
    benchmark_risk_latency()
