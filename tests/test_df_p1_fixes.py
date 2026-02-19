"""
🔬 SOVEREIGN SHIELD — P1 Fixes Verification (DF-B5, DF-C8, DF-D10)
Tests for:
  DF-B5: Target Clipping (prevents weight explosion)
  DF-D10: Concept Drift Detection (detects regime change)
  DF-C8: Rate Limiter Sync (syncs with server headers)
"""
import sys
import os
import unittest
import numpy as np
import time
from unittest.mock import MagicMock

# FORCE TESTNET FOR VERIFICATION
os.environ['BINANCE_USE_TESTNET'] = 'True'
os.environ['BINANCE_USE_DEMO'] = 'True'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.online_learning import OnlineLearner, DriftDetector
from utils.token_bucket import NumbaTokenBucket

class TestP1Fixes(unittest.TestCase):

    def test_df_b5_target_clipping(self):
        """[DF-B5] Verify that outlier rewards are clipped."""
        # Initialize learner with small clip
        clip = 5.0
        learner = OnlineLearner(learning_rate=0.1, target_clip=clip)
        
        weights = np.zeros(5, dtype=np.float32)
        state = np.ones(5, dtype=np.float32)
        next_state = np.ones(5, dtype=np.float32)
        
        # 1. Normal reward
        reward_normal = 1.0
        w1 = weights.copy()
        err1 = learner.learn_single(w1, state, 0.0, reward_normal, next_state)
        # Target=1.0. Error=1.0.
        
        # 2. Outlier reward (Massive)
        reward_massive = 1000.0 
        w2 = weights.copy()
        # Should be clipped to 5.0
        # Target=5.0. Error=5.0.
        err2 = learner.learn_single(w2, state, 0.0, reward_massive, next_state)
        
        print(f"✅ DF-B5: Normal Error={err1:.4f}, Massive Reward Error (Clipped)={err2:.4f}")
        
        # Verify err2 is close to target_clip (5.0) not 1000.0
        self.assertLess(abs(err2 - clip), 0.1, f"Error {err2} not clipped to {clip}")
        self.assertLess(w2[0], 1.0) 
        
    def test_df_d10_concept_drift(self):
        """[DF-D10] Verify detection of sudden error spikes."""
        # Window 200 > 100 for init.
        detector = DriftDetector(window_size=200, threshold=2.0)
        
        # 1. Establish baseline (low error ~0.1)
        print("✅ DF-D10: Warming up baseline...")
        for _ in range(150):
            err = 0.1 + np.random.normal(0, 0.01)
            detector.update(err)
            
        self.assertTrue(detector.initialized, "DriftDetector failed to initialize")
        base_mean = detector.baseline_mean
        print(f"   Baseline Mean: {base_mean:.4f}")
        
        # 2. Introduce Drift (Error spikes to 5.0 - Huge)
        print("   Injecting Drift...")
        triggered = False
        for i in range(20):
            drift_err = 5.0 + np.random.normal(0, 0.01)
            if detector.update(drift_err):
                print(f"✅ DRIFT DETECTED at step {i} (Error={drift_err:.4f})")
                triggered = True
                break
        
        self.assertTrue(triggered, "DriftDetector failed to trigger on huge error spike")

    def test_df_c8_rate_limiter_sync(self):
        """[DF-C8] Verify token bucket syncs with server used-weight."""
        rate = 10.0
        capacity = 100.0
        bucket = NumbaTokenBucket(rate, capacity)
        
        # 1. Initial State: Full (100 tokens)
        self.assertAlmostEqual(bucket.state[0], 100.0, places=1)
        
        # 2. Server says we used 90 (Weight=90).
        # Means we should have 10 tokens left.
        headers_usage = 90.0
        bucket.sync(headers_usage)
        
        # Verify sync
        tokens_after = bucket.state[0]
        print(f"✅ DF-C8: Used=90 → Tokens={tokens_after:.2f} (Expected ~10)")
        self.assertLess(tokens_after, 15.0)
        self.assertGreater(tokens_after, 5.0)
        
        # 3. Consume should now fail or wait
        allowed, wait = bucket.consume(20.0)
        self.assertFalse(allowed)
        
    def test_df_c8_sync_conservative(self):
        """[DF-C8] Verify we keep local state if it's MORE conservative than server."""
        rate = 10.0
        capacity = 100.0
        bucket = NumbaTokenBucket(rate, capacity)
        
        # 1. Drain locally to 10 tokens
        allowed, wait = bucket.consume(90.0)
        self.assertTrue(allowed, f"Consume(90) failed! Wait={wait}")
        self.assertLess(bucket.state[0], 11.0, f"State not updated: {bucket.state[0]}")
        
        print(f"DEBUG: Before Sync - Tokens: {bucket.state[0]}")
        
        # 2. Server says we used 0 (Fresh window? or lag).
        # Server implies we have 100 tokens.
        headers_usage = 0.0
        bucket.sync(headers_usage)
        
        tokens_after = bucket.state[0]
        print(f"DEBUG: After Sync - Tokens: {tokens_after}")
        
        print(f"✅ DF-C8: Local=10, Server=100 → Tokens={tokens_after:.2f} (Kept Conservative)")
        self.assertLess(tokens_after, 15.0)

if __name__ == '__main__':
    print("=" * 60)
    print("🔬 P1 FIXES VERIFICATION (DF-B5, DF-C8, DF-D10)")
    print("=" * 60)
    unittest.main(verbosity=2)
