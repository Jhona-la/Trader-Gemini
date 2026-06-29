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
        self.signals = []
    def put_nowait(self, item):
        self.signals.append(item)
    def put(self, item):
        self.signals.append(item)
    def get(self):
        if not self.signals: return None
        return self.signals.pop(0)
    def clear(self):
        self.signals = []

def benchmark_multisymbol_scaling():
    print("\n🚀 Starting Phase 58: Multi-Symbol Scaling Audit (20 symbols burst)...")
    
    n_symbols = 20
    symbols = [f"SYM_{i}/USDT" for i in range(n_symbols)]
    
    # 1. Setup Fleet
    print(f"   - Initializing fleet of {n_symbols} organisms...")
    strategies = []
    risk_manager = RiskManager()
    risk_manager.portfolio = None
    events_queue = MockQueue()
    data_provider = MagicMock()
    
    # Mock data (30 bars)
    n_bars = 50
    data = np.zeros(n_bars, dtype=[
        ('timestamp', 'M8[ms]'),
        ('open', 'f4'), ('high', 'f4'), ('low', 'f4'), 
        ('close', 'f4'), ('volume', 'f4')
    ])
    data['close'] = np.random.randn(n_bars).astype(np.float32) + 1000
    data['volume'] = 100
    
    data_provider.get_latest_bars.return_value = data
    data_provider.get_active_positions.return_value = {}

    for sym in symbols:
        gene = Genotype(symbol=sym)
        gene.init_brain(25, 4)
        # Inject fast path genes
        gene.genes.update({
            'adx_threshold': 20, 'strength_threshold': 0.6,
            'tp_pct': 0.015, 'sl_pct': 0.02,
            'long_mean_rev': 30, 'short_mean_rev': 70,
            'brain_weights': np.random.randn(25 * 4).astype(np.float32).tolist()
        })
        strat = HybridScalpingStrategy(data_provider, events_queue, gene)
        strategies.append(strat)

    # 2. Warm-up
    print("   - Warming up kernels...")
    for strat in strategies:
        me = MarketEvent(symbol=strat.symbol, close_price=1000.0)
        strat.calculate_signals(me)
    events_queue.clear()

    # 3. Measurement Loop (Burst Analysis)
    n_iterations = 1000
    burst_latencies = []
    
    print(f"   - Executing {n_iterations} burst cycles...")
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        
        # Simulating a market-wide "tick" where all symbols receive data
        for strat in strategies:
            me = MarketEvent(symbol=strat.symbol, close_price=1000.0)
            # Strategy Path
            strat.calculate_signals(me)
            
            # Risk Path (Process all signals generated in this tick)
            while True:
                sig = events_queue.get()
                if not sig: break
                risk_manager.generate_order(sig, 1000.0)
                
        end = time.perf_counter_ns()
        burst_latencies.append(end - start)

    avg_burst_us = np.mean(burst_latencies) / 1000
    p99_burst_us = np.percentile(burst_latencies, 99) / 1000
    per_symbol_us = avg_burst_us / n_symbols
    
    print("\n📈 SCALING METRICS (20 Symbol Fleet):")
    print(f"   - Total Fleet Burst Latency (Avg): {avg_burst_us:.2f} μs")
    print(f"   - Total Fleet Burst Latency (P99): {p99_burst_us:.2f} μs")
    print(f"   - Average Latency Per Symbol:      {per_symbol_us:.2f} μs")
    
    # Total burst should be well under 1ms
    if avg_burst_us < 500:
        print(f"\n✅ SUCCESS: Fleet burst latency is ultra-low ({avg_burst_us:.2f} μs)")
    else:
        print(f"\n❌ FAILED: Fleet latency exceeds 500μs ({avg_burst_us:.2f} μs)")

if __name__ == "__main__":
    benchmark_multisymbol_scaling()
