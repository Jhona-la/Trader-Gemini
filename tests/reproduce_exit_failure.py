
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime, timezone
import queue
from core.portfolio import Portfolio
from core.events import EventType, SignalType, SignalEvent

class MockDataProvider:
    def __init__(self):
        self.bars = []
    
    def get_latest_bars(self, symbol, n=1):
        return self.bars[-n:]

class TestExitFailure(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(initial_capital=1000.0, auto_save=False)
        self.data_provider = MockDataProvider()
        self.events_queue = queue.Queue()
        
    def test_high_volatility_exit(self):
        symbol = "BTC/USDT"
        entry_price = 50000.0
        
        # 1. Setup Position
        self.portfolio.positions[symbol] = {
            'quantity': 0.1,
            'avg_price': entry_price,
            'current_price': entry_price,
            'high_water_mark': entry_price
        }
        
        # 2. Simulate +6% Price Spike (Should Trigger TP > 0.8%)
        current_price = 53000.0 # +6%
        self.portfolio.positions[symbol]['current_price'] = current_price
        
        # 3. Running Check Exits
        print(f"\n🧪 SIMULATION: Price spiked to ${current_price} (+6.0%)")
        self.portfolio.check_exits(self.data_provider, self.events_queue)
        
        # 4. Verify Signal Generation
        if not self.events_queue.empty():
            event = self.events_queue.get()
            print(f"✅ SUCCESS: Generated Signal {event.signal_type} for {event.symbol}")
            self.assertEqual(event.type, EventType.SIGNAL)
            self.assertEqual(event.signal_type, SignalType.EXIT)
        else:
            print("❌ FAILURE: No Exit Signal generated despite +6% gain!")
            self.fail("Portfolio failed to catch +6% profit")

if __name__ == '__main__':
    unittest.main()
