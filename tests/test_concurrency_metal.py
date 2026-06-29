import unittest
import threading
import numpy as np
import time
from utils.rl_buffer import NumbaExperienceBuffer

class TestConcurrencyMetal(unittest.TestCase):
    def test_buffer_race(self):
        """Verify Spinlock prevents race conditions on Buffer Push"""
        print("\n🛡️ Testing Concurrency Guard (Spinlock)...")
        
        capacity = 2000
        state_dim = 4
        buf = NumbaExperienceBuffer(capacity, state_dim)
        
        n_threads = 10
        pushes_per_thread = 100
        
        def worker(thread_id):
            for i in range(pushes_per_thread):
                s = np.full(state_dim, float(thread_id))
                a = 1.0
                r = 1.0
                sn = s.copy()
                buf.push(s, a, r, sn)
                # Small sleep to induce context switch
                if i % 10 == 0:
                    time.sleep(0.0001)
                    
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Verification
        # Total writes = 10 * 100 = 1000
        # Capacity = 2000
        # Check size
        expected_size = n_threads * pushes_per_thread
        print(f"   > Expected Size: {expected_size}")
        print(f"   > Actual Size: {len(buf)}")
        
        self.assertEqual(len(buf), expected_size, f"Race Condition detected! Lost writes. {len(buf)} != {expected_size}")
        print("✅ Spinlock verified. No lost writes.")

if __name__ == '__main__':
    unittest.main()
