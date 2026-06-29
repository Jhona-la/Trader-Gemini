
import time
import queue
import logging
import statistics
from core.engine import Engine, BoundedQueue
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import OrderSide, SignalType, OrderType
from datetime import datetime, timezone

# Setup dummy components
class MockDataHandler:
    def process_event(self, event): pass
    def get_latest_bars(self, symbol, n=1): return None

class MockPortfolio:
    def __init__(self):
        self.positions = {}
        self.global_regime = 'STABLE'
        self.global_regime_data = {}
    def update_market_price(self, symbol, price): pass
    def check_exits(self, data_handler, events): pass
    def update_signal(self, event): pass
    def update_fill(self, event): pass
    def get_total_equity(self): return 1000.0

class MockRiskManager:
    def __init__(self):
        self.current_regime = 'STABLE'
    def check_stops(self, p, d): return []
    def update_equity(self, e): pass
    def generate_order(self, signal, price): 
        return OrderEvent(symbol=signal.symbol, order_type=OrderType.LIMIT, quantity=0.1, direction=OrderSide.BUY, price=price)
    def update_global_regime(self, regime): self.current_regime = regime

class MockExecutor:
    def execute_order(self, e): pass
    def cancel_order(self, s, o): return True

class MockStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.market_context = {}
    def calculate_signals(self, event): pass

def run_stress_test(event_count=1000, burst_size=50):
    events_q = BoundedQueue(maxsize=10000)
    engine = Engine(events_q)
    
    # Register mock handlers
    engine.register_data_handler(MockDataHandler())
    engine.register_portfolio(MockPortfolio())
    engine.register_risk_manager(MockRiskManager())
    engine.register_execution_handler(MockExecutor())
    engine.register_strategy(MockStrategy("BTC/USDT"))
    
    latencies = []
    
    print(f"🚀 Starting Stress Test: {event_count} events (Burst: {burst_size})")
    
    start_total = time.perf_counter()
    
    for i in range(event_count):
        # Create a mix of events (Phase 50: HFT Simulation)
        if i % 4 == 0:
            event = MarketEvent(symbol="BTC/USDT", timestamp=datetime.now(timezone.utc), close_price=90000.0)
        elif i % 4 == 1:
            event = SignalEvent(strategy_id="TEST", symbol="BTC/USDT", datetime=datetime.now(timezone.utc), signal_type=SignalType.LONG)
        elif i % 4 == 2:
            event = OrderEvent(symbol="BTC/USDT", order_type=OrderType.LIMIT, quantity=0.1, direction=OrderSide.BUY, price=90000.0)
        else:
            event = FillEvent(timeindex=datetime.now(timezone.utc), symbol="BTC/USDT", exchange="BINANCE", 
                              quantity=0.1, direction=OrderSide.BUY, fill_cost=9000, fill_price=90000, order_id="OID_123")
            
        # Put in queue
        events_q.put(event)
        
        # Process in bursts
        if i % burst_size == 0 or i == event_count - 1:
            burst_start = time.perf_counter()
            processed = 0
            while not events_q.empty():
                evt = events_q.get()
                engine.process_event(evt)
                processed += 1
            burst_end = time.perf_counter()
            latencies.append((burst_end - burst_start) / processed if processed > 0 else 0)

    end_total = time.perf_counter()
    duration = end_total - start_total
    throughput = event_count / duration
    
    avg_latency = statistics.mean(latencies) * 1000 # ms
    p95_latency = statistics.quantiles(latencies, n=20)[18] * 1000 # p95 in ms
    
    print("\n" + "="*40)
    print("📊 STRESS TEST RESULTS (GOD MODE)")
    print("="*40)
    print(f"Total Duration:   {duration:.4f}s")
    print(f"Throughput:       {throughput:.2f} events/s")
    print(f"Avg Latency:      {avg_latency:.4f}ms")
    print(f"P95 Latency:      {p95_latency:.4f}ms")
    print("="*40)
    
    if throughput > 500 and avg_latency < 0.1:
        print("✅ PERFORMANCE CERTIFIED (Institutional Tier)")
    else:
        print("⚠️ PERFORMANCE DEGRADED (Check Lock Contention)")

if __name__ == "__main__":
    run_stress_test(event_count=5000)
