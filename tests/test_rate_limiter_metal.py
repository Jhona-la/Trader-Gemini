import unittest
import time
from utils.token_bucket import NumbaTokenBucket

class TestRateLimiterMetal(unittest.TestCase):
    def test_bucket_mechanics(self):
        """Verify Token Bucket Logic"""
        print("\n🛡️ Testing Numba Token Bucket...")
        
        rate = 10.0 # 10 tokens / sec
        capacity = 20.0 # Max burst 20
        bucket = NumbaTokenBucket(rate, capacity)
        
        # 1. Burst Consumption
        # Should allow 20 tokens immediately
        success, wait = bucket.consume(20.0)
        self.assertTrue(success, "Burst failed")
        self.assertEqual(wait, 0.0)
        
        # 2. Block immediately after burst
        success, wait = bucket.consume(1.0)
        self.assertFalse(success, "Bucket should be empty")
        self.assertGreater(wait, 0.0)
        print(f"   > Blocked correctly. Wait time: {wait:.3f}s")
        
        # 3. Refill
        # Wait for 0.1s -> should gain 1 token (10 * 0.1)
        time.sleep(0.12) 
        
        success, wait = bucket.consume(1.0)
        self.assertTrue(success, "Refill failed")
        print("   > Refill verified.")
        
    def test_concurrency(self):
        """Verify Thread Safety"""
        import threading
        
        rate = 1000.0
        capacity = 2000.0
        bucket = NumbaTokenBucket(rate, capacity) # Starts full
        
        n_threads = 10
        ops = 100
        
        def worker():
            for _ in range(ops):
                bucket.consume(1.0)
                
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        # Should have consumed 1000 tokens
        # Remaining should be ~1000 (plus slight refill during execution)
        
        remaining = bucket.state[0]
        print(f"   > Remaining Tokens: {remaining:.2f} (Expected ~1000)")
        self.assertTrue(900 < remaining < 1100, "Concurrency accounting error")
        print("✅ Token Bucket verified.")

if __name__ == '__main__':
    unittest.main()
