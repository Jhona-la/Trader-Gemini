
import time
import sys
import os
import threading
import queue
import statistics

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import Engine, MarketEvent
from core.events import EventType

def profile_engine():
    print("🚀 Starting Engine Latency Profiler...")
    
    # Setup
    q = queue.Queue()
    engine = Engine(events_queue=q)
    
    # Mock components to isolate Engine overhead
    class MockStrategy:
        def __init__(self):
            self.symbol = "BTC/USDT"
        def calculate_signals(self, event):
            pass # No-op to measure pure engine overhead

    engine.register_strategy(MockStrategy())
    
    # Start Engine in Thread
    engine_thread = threading.Thread(target=engine.run, daemon=True)
    engine_thread.start()
    
    # Warmup
    print("🔥 Warming up...")
    for _ in range(100):
        q.put(MarketEvent(symbol="BTC/USDT", close_price=50000.0))
    time.sleep(1)
    
    # Benchmark
    ITERATIONS = 5000
    latencies = []
    
    print(f"⏱️  Benchmarking {ITERATIONS} events...")
    
    for i in range(ITERATIONS):
        start = time.perf_counter()
        
        # Inject Event
        evt = MarketEvent(symbol="BTC/USDT", close_price=50000.0 + i)
        q.put(evt)
        
        # Measurement is tricky for async consumer
        # We rely on Engine metrics if available, or we measure put throughput
        # But to measure processing time, we need to instrument the engine or use internal metrics
        
        # Since we optimized Engine to have metrics, let's use them!
        
        # Wait for queue to drain (approx)
        while not q.empty():
            pass
            
    # Stop
    engine.running = False
    
    # Retrieve Metrics from Engine
    # Note: Engine metrics might need to be accessed cleanly
    metrics = engine.metrics
    
    print("\n📊 RESULTS:")
    print(f"Total Events: {metrics['processed_events']}")
    print(f"Total Errors: {metrics['errors']}")
    
    # If Internal metrics aren't detailed enough, we calculate throughput
    # Throughput = Events / Second
    
    # Let's do a throughput test instead
    engine.running = True
    q = queue.Queue()
    engine = Engine(events_queue=q)
    engine.register_strategy(MockStrategy())
    t_start = threading.Thread(target=engine.run, daemon=True)
    t_start.start()
    
    start_time = time.perf_counter()
    for _ in range(10000):
        q.put(MarketEvent(symbol="BTC/USDT", close_price=50000.0))
    
    while engine.metrics['processed_events'] < 10000:
        time.sleep(0.001)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    throughput = 10000 / duration
    latency_per_event_ms = (duration / 10000) * 1000
    
    print(f"⚡ Throughput: {throughput:.2f} events/sec")
    print(f"⚡ Avg Latency: {latency_per_event_ms:.4f} ms/event")
    
    if latency_per_event_ms < 10.0:
        print("✅ PASS: Latency < 10ms")
    else:
        print("❌ FAIL: Latency > 10ms")

if __name__ == "__main__":
    profile_engine()
