import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database import DatabaseHandler
import threading

def run_test():
    db = DatabaseHandler("test_async.db")
    
    print("Main Thread ID:", threading.get_ident())
    start = time.perf_counter_ns()
    
    # Encolar 1000 operaciones
    for i in range(1000):
        db.log_error("test_module", f"Async test error {i}", "INFO")
        
    end = time.perf_counter_ns()
    print(f"Time to queue 1000 events: {(end - start)/1000:,.0f} ns/event")
    
    # Wait for background thread to process
    time.sleep(1)
    
    # Verify records were inserted
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM errors")
    count = cursor.fetchone()[0]
    print(f"Records successfully inserted in background: {count}")
    
    db.close()
    
if __name__ == "__main__":
    run_test()
