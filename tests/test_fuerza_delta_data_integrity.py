import unittest
import os
import shutil
import threading
import time
import pandas as pd
from utils.data_manager import safe_append_csv, safe_read_csv

class TestF43_DataIntegrity(unittest.TestCase):
    def setUp(self):
        self.test_csv = "tests/temp_trades_f43.csv"
        if os.path.exists(self.test_csv):
            try:
                os.remove(self.test_csv)
            except:
                pass
            
    def tearDown(self):
        if os.path.exists(self.test_csv):
            try:
                os.remove(self.test_csv)
            except:
                pass

    def test_concurrent_read_write(self):
        """
        Verify that multiple threads writing/reading simultaneously do not cause PermissionError
        and data is preserved correctly.
        """
        # Create initial file
        initial_data = {'id': 'init', 'val': 0}
        safe_append_csv(self.test_csv, initial_data)
        
        errors = []
        write_count = 50
        read_count = 50
        
        def writer_thread():
            for i in range(write_count):
                try:
                    safe_append_csv(self.test_csv, {'id': f'w_{i}', 'val': i})
                    time.sleep(0.001) # Small delay to mix operations
                except Exception as e:
                    errors.append(f"Writer error: {e}")
                    
        def reader_thread():
            for i in range(read_count):
                try:
                    df = safe_read_csv(self.test_csv)
                    if df is None:
                        # Depending on implementation, returning None on lock contention might be valid or invalid.
                        # safe_read_csv returns None if file doesn't exist or exception.
                        # File exists from start. Exception = bad.
                        errors.append("Reader got None (Access Error?)")
                    else:
                        # Naive check: should be valid dataframe
                        pass
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(f"Reader error: {e}")
                    
        t1 = threading.Thread(target=writer_thread)
        t2 = threading.Thread(target=reader_thread)
        t3 = threading.Thread(target=writer_thread) # Another writer
        
        t1.start()
        t2.start()
        t3.start()
        
        t1.join()
        t2.join()
        t3.join()
        
        if errors:
            self.fail(f"Concurrent errors: {errors[:5]}")
            
        # Verify final count
        # 1 init + 50 writes (t1) + 50 writes (t3) = 101 rows
        df = safe_read_csv(self.test_csv)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1 + write_count * 2)
        print(f"✅ F43 Concurrency Test Passed: {len(df)} rows written safely.")

if __name__ == '__main__':
    unittest.main()
