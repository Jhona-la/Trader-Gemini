
import time
from core.orderbook import OrderBook

def test_orderbook():
    print("🧪 Testing OrderBook Performance...")
    ob = OrderBook(max_depth=100)
    
    start = time.perf_counter()
    # Simulate 100k updates
    for i in range(100000):
        ob.update_bid(100.0 + (i % 100) * 0.1, 1.0)
        ob.update_ask(105.0 + (i % 100) * 0.1, 1.0)
    
    duration = time.perf_counter() - start
    print(f"✅ 100k Updates took: {duration:.4f}s")
    
    snap = ob.get_snapshot()
    print(f"Snapshot Bids: {len(snap['bids'])}, Asks: {len(snap['asks'])}")

if __name__ == "__main__":
    test_orderbook()
