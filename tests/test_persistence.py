
import unittest
import os
import sys
import shutil
import time

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_manager import DatabaseHandler

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.db_path = "tests/test_crash.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = DatabaseHandler(self.db_path)
        
    def tearDown(self):
        self.db.close()
        # Wait for file unlock
        time.sleep(0.1)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
    def test_crash_recovery(self):
        print("\n🧪 Testing Crash Recovery...")
        
        # 1. Open Position
        print("   -> Opening position BTC/USDT...")
        self.db.log_fill_event_atomic(
            trade_payload={
                'symbol': 'BTC/USDT', 'side': 'BUY', 'quantity': 0.1, 
                'price': 50000.0, 'timestamp': '2024-01-01 12:00:00'
            },
            position_payload={
                'symbol': 'BTC/USDT', 'quantity': 0.1, 'entry_price': 50000.0,
                'current_price': 50000.0, 'pnl': 0.0
            }
        )
        
        # 2. Simulate Crash (Hard Close)
        self.db.close()
        print("   -> 💥 SYSTEM CRASH SIMULATED (DB Closed)")
        
        # 3. Recover
        print("   -> 🚑 Restarting System...")
        new_db = DatabaseHandler(self.db_path)
        positions = new_db.get_open_positions()
        
        # 4. Verify
        self.assertTrue('BTC/USDT' in positions, "Position missing after recovery")
        self.assertEqual(positions['BTC/USDT']['quantity'], 0.1, "Quantity mismatch")
        print("   ✅ Recovery Successful! Position restored.")
        new_db.close()

if __name__ == '__main__':
    unittest.main()
