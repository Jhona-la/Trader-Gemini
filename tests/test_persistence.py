import unittest
import os
import sqlite3
import shutil
from datetime import datetime, timezone
from data.database import DatabaseHandler
from core.portfolio import Portfolio
from config import Config

class TestPersistence(unittest.TestCase):
    def setUp(self):
        # Use a test database
        self.test_db = "test_trader_gemini.db"
        self.db_path = os.path.join(Config.DATA_DIR, self.test_db)
        # Ensure clean state
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.db = DatabaseHandler(db_name=self.test_db)
        
    def tearDown(self):
        self.db.close()
        # Clean up
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.db_path + "-wal"):
            os.remove(self.db_path + "-wal")
        if os.path.exists(self.db_path + "-shm"):
            os.remove(self.db_path + "-shm")

    def test_wal_mode_enabled(self):
        """Rule 5.1: Verify WAL mode is active"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        self.assertEqual(mode.upper(), 'WAL', "Database should be in WAL mode")

    def test_atomic_persistence_and_recovery(self):
        """Rule 5.2 & 5.3: Verify Atomic Write and Crash Recovery"""
        # 1. Simulate Atomic Fill
        trade_payload = {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'quantity': 0.1,
            'price': 50000.0,
            'timestamp': datetime.now(timezone.utc),
            'pnl': 0.0,
            'commission': 5.0
        }
        position_payload = {
            'symbol': 'BTC/USDT',
            'quantity': 0.1,
            'entry_price': 50000.0,
            'current_price': 50000.0,
            'pnl': 0.0
        }
        
        self.db.log_fill_event_atomic(trade_payload, position_payload)
        
        # 2. Verify Data in DB
        conn = self.db.get_connection()
        pos = conn.execute("SELECT * FROM positions WHERE symbol='BTC/USDT'").fetchone()
        self.assertIsNotNone(pos)
        self.assertEqual(pos['quantity'], 0.1)
        
        trade = conn.execute("SELECT * FROM trades WHERE symbol='BTC/USDT'").fetchone()
        self.assertIsNotNone(trade)
        self.assertEqual(trade['price'], 50000.0)
        
        # 3. Simulate Restart (New Portfolio using same DB)
        # Mocking Config to point to test DB indirectly or just using the handlers
        # Since Portfolio initializes its own DB handler using default name, 
        # we need to subclass or patch it to use our test DB.
        
        # We will manually test the recovery logic using the DB handler we already have
        # simulating what Portfolio.restore_state_from_db does.
        
        recovered_positions = self.db.get_open_positions()
        self.assertIn('BTC/USDT', recovered_positions)
        self.assertEqual(recovered_positions['BTC/USDT']['quantity'], 0.1)
        
    def test_close_position_persistence(self):
        """Verify closing a position removes it from active set"""
        # Open
        self.db.update_position('ETH/USDT', 1.0, 3000.0, 3000.0, 0.0)
        
        # Verify Open
        positions = self.db.get_open_positions()
        self.assertIn('ETH/USDT', positions)
        
        # Atomic Close
        trade_payload = {
            'symbol': 'ETH/USDT',
            'side': 'SELL',
            'quantity': 1.0, 
            'price': 3100.0,
            'timestamp': datetime.now(timezone.utc),
            'pnl': 100.0,
            'commission': 3.1
        }
        position_payload = {
            'symbol': 'ETH/USDT',
            'quantity': 0.0, # CLOSED
            'entry_price': 3000.0,
            'current_price': 3100.0,
            'pnl': 0.0
        }
        
        self.db.log_fill_event_atomic(trade_payload, position_payload)
        
        # Verify Closed (Removed from DB positions table)
        positions = self.db.get_open_positions()
        self.assertNotIn('ETH/USDT', positions)
        
        # Verify Trade Logged
        conn = self.db.get_connection()
        trades = conn.execute("SELECT * FROM trades WHERE symbol='ETH/USDT'").fetchall()
        self.assertTrue(len(trades) >= 1)

if __name__ == '__main__':
    unittest.main()
