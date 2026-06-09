
import unittest
import pandas as pd
import numpy as np
import os
import sys
# Add root to path so we can import strategies
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
from data.database import DatabaseHandler
from execution.liquidity_guardian import LiquidityGuardian
from strategies.technical import HybridScalpingStrategy

class TestGodMode(unittest.TestCase):
    
    def test_phase_19_float32(self):
        """Verify that Technical Strategy provides lightweight DataFrames."""
        print("\nTesting Phase 19: Float32 Optimization...")
        strategy = HybridScalpingStrategy(None, None)
        
        # Create Dummy Data (Structured Numpy Array to match production recarray format)
        dtype_spec = [('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8'), ('timestamp', 'O')]
        df = np.zeros(100, dtype=dtype_spec)
        df['high'] = np.random.rand(100)
        df['low'] = np.random.rand(100)
        df['close'] = np.random.rand(100)
        df['volume'] = np.random.rand(100)
        df['timestamp'] = pd.date_range(start='2026-01-01', periods=100)
        
        processed_df = strategy.calculate_indicators(df)
        
        dtype = processed_df['rsi'].dtype
        print(f"RSI Dtype: {dtype}")
        self.assertTrue(str(dtype) == 'float32', f"Expected float32, got {dtype}")

    def test_phase_43_db_healing(self):
        """Verify Database Auto-Healing logic."""
        print("\nTesting Phase 43: DB Auto-Healing...")
        from config import Config
        db_name = "test_god_mode.db"
        db_path = os.path.join(Config.DATA_DIR, db_name)
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass
        
        db = DatabaseHandler(db_name)
        db.create_tables()
        self.assertTrue(db.check_integrity())
        
        # Corrupt the DB manually (close connection first to avoid Windows file locks)
        if db.conn:
            db.conn.close()
            db.conn = None
        with open(db.db_path, 'wb') as f:
            f.write(b'CORRUPTED_DATA_HEADER')
            
        # Check integrity again (should fail and heal)
        print("Corrupting DB and checking integrity...")
        is_healed = db.check_integrity() # Should trigger heal
        
        # Verify it rotated
        self.assertFalse(is_healed, "First check should return False (it was corrupt)")
        
        # Re-check newly created DB
        print("Verifying healed DB...")
        db2 = DatabaseHandler(db_name)
        self.assertTrue(db2.check_integrity(), "Healed DB should be valid")
        
        # Clean up
        db2.close()
        for f in os.listdir(Config.DATA_DIR):
            if f.startswith(db_name):
                try: os.remove(os.path.join(Config.DATA_DIR, f))
                except: pass

    def test_phase_33_spoofing(self):
        """Verify Spoofing Detection."""
        print("\nTesting Phase 33: Anti-Spoofing...")
        guardian = LiquidityGuardian(None)
        
        # 1. Big Wall appears
        bids_t1 = [[100, 10.0]] 
        asks_t1 = [[101, 10.0]]
        guardian._detect_spoofing('BTC/USDT', bids_t1, asks_t1)
        
        # 2. Wall Vanishes instantly (Volume drops to 0.1)
        bids_t2 = [[100, 0.1]]
        asks_t2 = [[101, 10.0]]
        
        is_spoofing, reason = guardian._detect_spoofing('BTC/USDT', bids_t2, asks_t2)
        print(f"Spoofing Result: {is_spoofing} | Reason: {reason}")
        
        self.assertTrue(is_spoofing)
        self.assertIn("Spoofing detected", reason)

if __name__ == '__main__':
    unittest.main()
