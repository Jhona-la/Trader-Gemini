import sys
import os
import time
import queue
from datetime import datetime, timezone

sys.path.append(os.getcwd())

from core.engine import Engine, BoundedQueue
from core.events import SignalEvent, SignalType

def test_engine_improvements():
    print("Testing Engine Improvements...")
    
    # 1. Bounded Queue
    bq = BoundedQueue(maxsize=3)
    bq.put(1)
    bq.put(2)
    bq.put(3)
    try:
        bq.put(4)
        print("BoundedQueue discarded oldest item correctly (no blocker).")
    except queue.Full:
        print("FAIL: BoundedQueue blocked!")
        
    assert bq.qsize() == 3
    item = bq.get()
    print(f"First item popped: {item}")
    assert item == 2 # 1 was discarded
    
    # 2. Engine Initialization
    engine = Engine()
    print("Engine initialized successfully.")
    
    # 3. Helper Logic Check
    # Mock Risk Manager
    class MockRM:
        current_regime = 'TRENDING_BULL'
        
    engine.risk_manager = MockRM()
    regime = engine._get_current_market_regime()
    print(f"Detected Regime: {regime}")
    assert regime == 'TRENDING_BULL'
    
    print("✅ ENGINE TEST PASSED")

if __name__ == "__main__":
    test_engine_improvements()
