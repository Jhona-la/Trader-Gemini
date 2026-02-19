import time
import numpy as np
import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Root path
sys.path.append(os.getcwd())

from strategies.technical import HybridScalpingStrategy
from risk.risk_manager import RiskManager
from core.genotype import Genotype
from core.events import MarketEvent, SignalEvent
from core.enums import SignalType

class MockQueue:
    def __init__(self):
        self.item = None
    def put_nowait(self, item):
        self.item = item
    def put(self, item):
        self.item = item
    def get(self):
        val = self.item
        self.item = None
        return val

def benchmark_e2e_latency():
    print("\n🚀 Starting Phase 57: Extreme E2E Latency Benchmark (Tick-to-Order)...")
    
    # 1. Setup Components
    symbol = "BTC/USDT"
    genotype = Genotype(symbol=symbol)
    genotype.init_brain(25, 4) # Match fused kernel input dim
    # Inject missing genes required by HybridScalpingStrategy
    genotype.genes['adx_threshold'] = 20
    genotype.genes['strength_threshold'] = 0.6
    genotype.genes['tp_pct'] = 0.015
    genotype.genes['sl_pct'] = 0.02
    genotype.genes['brain_weights'] = np.random.randn(25 * 4).astype(np.float32).tolist()
    
    events_queue = MockQueue()
    data_provider = MagicMock()
    
    strategy = HybridScalpingStrategy(
        data_provider=data_provider,
        events_queue=events_queue,
        genotype=genotype
    )
    
    risk_manager = RiskManager()
    risk_manager.portfolio = None # Fast path
    
    # 2. Mock Data (30 bars)
    n = 50
    data = np.zeros(n, dtype=[
        ('timestamp', 'M8[ms]'),
        ('open', 'f4'), ('high', 'f4'), ('low', 'f4'), 
        ('close', 'f4'), ('volume', 'f4')
    ])
    data['close'] = np.random.randn(n).astype(np.float32) + 10000
    data['high'] = data['close'] + 10
    data['low'] = data['close'] - 10
    data['volume'] = 1000
    
    data_provider.get_latest_bars.return_value = data
    data_provider.get_active_positions.return_value = {}
    
    # 3. Warm-up (Compiling Numba Kernels)
    print("   - Warming up kernels...")
    market_event = MarketEvent(symbol=symbol, close_price=10000.0)
    strategy.calculate_signals(market_event)
    signal = events_queue.get()
    if signal:
        risk_manager.generate_order(signal, 10000.0)
    
    # 4. Measurement Loop
    n_iterations = 2000
    latencies = []
    
    print(f"   - Executing {n_iterations} iterations...")
    for _ in range(n_iterations):
        # Hot path start
        start = time.perf_counter_ns()
        
        # A. Strategy Path (Fused Kernel)
        strategy.calculate_signals(market_event)
        signal = events_queue.get()
        
        # B. Risk Path (In-memory Cache)
        if signal:
            risk_manager.generate_order(signal, 10000.0)
            
        end = time.perf_counter_ns()
        # Hot path end
        
        latencies.append(end - start)
        
    avg_ns = np.mean(latencies)
    p95_ns = np.percentile(latencies, 95)
    p99_ns = np.percentile(latencies, 99)
    std_ns = np.std(latencies)
    
    print("\n📈 PERFORMANCE METRICS (Tick-to-Order):")
    print(f"   - Average Latency: {avg_ns/1000:.2f} μs")
    print(f"   - P95 Latency:     {p95_ns/1000:.2f} μs")
    print(f"   - P99 Latency:     {p99_ns/1000:.2f} μs")
    print(f"   - StdDev (Jitter): {std_ns/1000:.2f} μs")
    
    if avg_ns < 500000:
        print(f"\n✅ SUCCESS: End-to-end latency below 500μs ({avg_ns/1000:.2f} μs)")
    else:
        print(f"\n❌ FAILED: Latency exceeds 500μs ({avg_ns/1000:.2f} μs)")

if __name__ == "__main__":
    benchmark_e2e_latency()
