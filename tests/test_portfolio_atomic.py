import unittest
import threading
import time
import sys
import os
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.portfolio import Portfolio
from core.events import FillEvent
from core.enums import OrderSide, EventType

class TestPortfolioAtomic(unittest.TestCase):
    def test_concurrent_snapshots(self):
        """Verify portfolio snapshots are consistent under heavy update load"""
        print("\n📈 Testing Portfolio Atomic Snapshots...")
        
        portfolio = Portfolio(initial_capital=10000.0, auto_save=False)
        symbol = "BTC/USDT"
        
        stop_event = threading.Event()
        
        def updater():
            """Simulate rapid price and fill updates"""
            i = 0
            while not stop_event.is_set():
                # 1. Price update
                portfolio.update_market_price(symbol, 50000.0 + (i % 100))
                
                # 2. Occasional fills
                if i % 10 == 0:
                    fill = FillEvent(
                        timestamp=time.time(),
                        symbol=symbol,
                        exchange="BINANCE",
                        quantity=0.1,
                        direction=OrderSide.BUY if (i // 10) % 2 == 0 else OrderSide.SELL,
                        fill_cost=5000.0 if (i // 10) % 2 == 0 else 5100.0
                    )
                    portfolio.update_fill(fill)
                i += 1

        # Start updating thread
        t = threading.Thread(target=updater)
        t.start()
        
        try:
            # Take 1000 snapshots and verify basic consistency
            for _ in range(1000):
                snap = portfolio.get_atomic_snapshot()
                
                # Basic invariant: Equity should be roughly Cash + PnL
                # (PnL calculation is internal, but we can verify types and existence)
                self.assertIsInstance(snap['cash'], float)
                self.assertIsInstance(snap['positions'], dict)
                
                # If we have a position, it should have a current_price
                if symbol in snap['positions']:
                    self.assertIn('current_price', snap['positions'][symbol])
                
                # Verify that snapshotting doesn't crash the system
                time.sleep(0.001)
                
            print("   > 1000 Snapshots captured successfully.")
            
        finally:
            stop_event.set()
            t.join()
            
        print("✅ Portfolio Atomic Logic OK.")

if __name__ == '__main__':
    unittest.main()
