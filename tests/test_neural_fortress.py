import unittest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sophia.rewards import reward_engine, TesisDecayReason
from ml.replay_buffer import PrioritizedReplayBuffer
from strategies.ml_strategy import MLStrategyHybridUltimate
from config import Config

class TestNeuralFortress(unittest.TestCase):
    
    def test_asymmetric_reward_shaping(self):
        """
        Verify that Drawdowns and Axioma penalties are punished asymmetrically.
        A trade that wins 1% but suffered a 2% drawdown should be punished.
        """
        # Case 1: Clean Win (1% PnL, 0% max drawdown)
        reward_clean = reward_engine.calculate_reward(
            pnl_pct=0.01, max_drawdown_pct=0.0, 
            axioma_diagnosis=TesisDecayReason.NONE, duration_seconds=60
        )
        self.assertAlmostEqual(reward_clean, 0.1) # 0.01 * 10
        
        # Case 2: Dirty Win (1% PnL, but 1.5% max drawdown)
        reward_dirty = reward_engine.calculate_reward(
            pnl_pct=0.01, max_drawdown_pct=-0.015, # drawdown is negative magnitude
            axioma_diagnosis=TesisDecayReason.NONE, duration_seconds=60
        )
        # Expected: 0.1 - (0.015 * 20) = 0.1 - 0.3 = -0.2
        self.assertAlmostEqual(reward_dirty, -0.2)
        
        # Case 3: Axioma Penalty (Structural failure)
        reward_axioma = reward_engine.calculate_reward(
            pnl_pct=-0.005, max_drawdown_pct=0.0,
            axioma_diagnosis=TesisDecayReason.THESIS_DECAY, duration_seconds=60
        )
        # Expected: (-0.005 * 10) + (-0.5) = -0.05 - 0.5 = -0.55
        self.assertAlmostEqual(reward_axioma, -0.55)

    def test_prioritized_replay_buffer(self):
        """
        Verify that Black Swans (huge losses or thesis decay) are sampled with
        higher probability than normal noise.
        """
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add 10 normal noise events (Reward ~ 0.0)
        for _ in range(10):
            buffer.add(
                state=np.array([0.5, 0.5, 0.5]), action=0.5, reward=0.01, 
                next_state=np.array([0.5, 0.5, 0.5]), log_prob=0.0, axioma_reason="NONE"
            )
            
        # Add 1 Black Swan event (Reward -1.5, reason THESIS_DECAY)
        buffer.add(
            state=np.array([0.1, 0.9, 0.1]), action=0.9, reward=-1.5, 
            next_state=np.array([0.1, 0.9, 0.1]), log_prob=-1.0, axioma_reason="THESIS_DECAY"
        )
        
        # Sample drastically to see if Black Swan appears frequently
        black_swan_sampled = False
        for _ in range(20):
            # Sample batch of 1
            batch, idxs, weights = buffer.sample(batch_size=1)
            if batch[0][5] == "THESIS_DECAY":
                black_swan_sampled = True
                break
                
        self.assertTrue(black_swan_sampled, "Black Swan should be heavily prioritized in PER buffer")

if __name__ == '__main__':
    unittest.main()
