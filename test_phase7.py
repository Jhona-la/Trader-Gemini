from core.engine import PriorityBoundedQueue
from core.events import MarketEvent
from risk.circuit_breaker import CircuitBreaker
from data.binance_loader import BinanceData
import numpy as np

def test_queue_triage():
    print("Testing PriorityBoundedQueue Backpressure...")
    queue = PriorityBoundedQueue()
    queue._items_count = 4500
    
    # Priority 2 event (should be dropped because items_count > 4000)
    ev_cosmetic = MarketEvent(symbol="BTC/USDT", close_price=100)
    ev_cosmetic.priority = 2
    ev_cosmetic.type = "COSMETIC"
    
    queue.put(ev_cosmetic)
    assert queue._items_count == 4500, "Event priority 2 should be dropped"
    
    queue._items_count = 4900
    ev_signal = MarketEvent(symbol="BTC/USDT", close_price=100)
    ev_signal.priority = 1
    ev_signal.type = "SIGNAL"
    
    queue.put(ev_signal)
    assert queue._items_count == 4901, "SIGNAL event should be kept"

def test_circuit_breaker():
    print("Testing CircuitBreaker...")
    cb = CircuitBreaker()
    cb.record_price("BTC/USDT", 50000)
    cb.record_price("BTC/USDT", 40000) # -20% drop (Flash Crash)
    assert not cb.check_health("BTC/USDT"), "CircuitBreaker should trip on flash crash"

def test_memory_view():
    print("Testing BinanceLoader memoryview...")
    res = np.empty(10, dtype=[('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'), ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')])
    view = res.view()
    view.flags.writeable = False
    try:
        view['open'][0] = 100.0
        assert False, "Should raise ValueError since view is read-only"
    except ValueError:
        print("Memoryview is read-only successfully")

if __name__ == "__main__":
    try:
        test_queue_triage()
        test_circuit_breaker()
        test_memory_view()
        print("All tests passed.")
    except Exception as e:
        print(f"Test failed: {e}")
