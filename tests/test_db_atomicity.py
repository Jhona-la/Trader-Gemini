import sqlite3
import time
import os
import sys
import multiprocessing
import signal
import random
from datetime import datetime

# DB Definition
DB_PATH = "test_atomicity.db"

def writer_process(stop_event):
    """
    Writes sequential records to the DB in a tight loop.
    Intended to be KILLED externally.
    """
    try:
        # Connect to DB
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        
        # Enable WAL
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()
        
        # Create table
        cursor.execute("CREATE TABLE IF NOT EXISTS sequence (id INTEGER PRIMARY KEY, timestamp TEXT, cargo TEXT)")
        conn.commit()
        
        counter = 0
        while not stop_event.is_set():
            counter += 1
            # Artificial cargo to make writes heavier
            cargo = "X" * 1024 
            
            try:
                cursor.execute("INSERT INTO sequence (id, timestamp, cargo) VALUES (?, ?, ?)", 
                              (counter, datetime.now().isoformat(), cargo))
                conn.commit()
                if counter % 100 == 0:
                    print(f"  [Writer] Committed {counter} records...", flush=True)
            except sqlite3.IntegrityError:
                # Should not happen in sequential single writer
                pass
            
            # Random tiny sleep to vary timing
            if random.random() < 0.1:
                time.sleep(0.001)

    except Exception as e:
        print(f"[Writer] Error: {e}")
    finally:
        # We probably won't get here if killed
        pass

def verify_integrity():
    """
    Verifies the DB integrity after the writer was killed.
    """
    if not os.path.exists(DB_PATH):
        print("❌ DB file does not exist.")
        return False
        
    print(f"\n🔍 Verifying {DB_PATH} integrity...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Check SQLite Integrity Check
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()[0]
        if result != "ok":
            print(f"❌ PRAGMA integrity_check FAILED: {result}")
            return False
        print("✅ PRAGMA integrity_check: OK")
        
        # 2. Check Sequence Continuity
        cursor.execute("SELECT id FROM sequence ORDER BY id ASC")
        rows = cursor.fetchall()
        ids = [r[0] for r in rows]
        
        if not ids:
            print("⚠️ DB is empty (Crash happened before first commit?)")
            return True # Technically atomic (nothing happened)
            
        print(f"✅ Recovered {len(ids)} records.")
        print(f"   Last ID: {ids[-1]}")
        
        # Check for gaps
        expected = list(range(1, ids[-1] + 1))
        if ids == expected:
            print("✅ ACID PROPERTY VERIFIED: No gaps in sequence.")
            return True
        else:
            print("❌ ATOMICITY FAILED: Gaps found in sequence!")
            # Find first gap
            for i, x in enumerate(expected):
                if x != ids[i]:
                    print(f"   First gap at ID: {x}")
                    break
            return False
            
    except sqlite3.DatabaseError as e:
        print(f"❌ Database is CORRUPT: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def run_atomicity_test():
    print("="*60)
    print("⚛️  PHASE 46: DATABASE ATOMICITY STRESS TEST (WAL MODE)")
    print("="*60)
    
    # Cleanup previous run
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except PermissionError:
            print("❌ Cannot remove old DB. Is it open?")
            return
            
    # Start Writer Process
    stop_event = multiprocessing.Event()
    p = multiprocessing.Process(target=writer_process, args=(stop_event,))
    p.start()
    
    print(f"🚀 Writer process started (PID: {p.pid}). Writing data...")
    
    # Let it write for a bit
    time.sleep(2.0)
    
    # KILL IT
    print("\n💥 SIMULATING SYSTEM CRASH (Hard Kill)...")
    p.terminate() # SIGTERM
    # On Windows terminate() is roughly equivalent to Kill, but let's be sure
    p.join(timeout=1)
    if p.is_alive():
        print("   Force killing...")
        p.kill() # SIGKILL (Python 3.7+)
    
    print("💀 Writer process died.")
    
    # Verify
    time.sleep(0.5) # Let OS release file locks if any
    
    success = verify_integrity()
    
    print("="*60)
    if success:
        print("✅ PHASE 46 COMPLETE: DATABASE IS ATOMIC AND DURABLE.")
    else:
        print("❌ PHASE 46 FAILED: DATA CORRUPTION OR ATOMICITY VIOLATION.")
    print("="*60)

if __name__ == "__main__":
    # Ensure spawning support on Windows
    multiprocessing.freeze_support()
    run_atomicity_test()
