import unittest
import numpy as np
from utils.rl_buffer import NumbaExperienceBuffer
from core.online_learning import OnlineLearner

class TestRLBuffer(unittest.TestCase):
    def test_buffer_mechanics(self):
        """Verify Numba Ring Buffer logic"""
        print("\n🧠 Testing RL Experience Buffer...")
        
        capacity = 100
        state_dim = 10
        buf = NumbaExperienceBuffer(capacity, state_dim)
        
        # 1. Fill Buffer
        for i in range(120): # Overflow by 20
            s = np.ones(state_dim) * i
            a = 1.0
            r = float(i)
            sn = np.ones(state_dim) * (i+1)
            buf.push(s, a, r, sn)
            
        self.assertEqual(len(buf), capacity, "Buffer size should be capped at capacity")
        
        # Verify Overwritting (Head should be at 20)
        # Last inserted was 119. 
        # Check if sample returns valid data
        batch_s, batch_a, _, _ = buf.sample(5)
        self.assertEqual(batch_s.shape, (5, state_dim))
        print("   > Buffer Push/Sample OK.")
        
    def test_jit_learning(self):
        """Verify JIT SGD Kernel updates"""
        print("⚡ Testing JIT SGD Kernel...")
        
        # Setup
        learner = OnlineLearner(learning_rate=0.01)
        buf = NumbaExperienceBuffer(50, 4) # 4 inputs
        
        # Create a linear pattern: Reward = sum(State)
        # Ideal weights = [1, 1, 1, 1] for Action 0
        state = np.array([1.0, 0.5, 0.5, 1.0], dtype=np.float32)
        action = 0.0
        reward = 3.0 # 1+0.5+0.5+1
        next_state = np.zeros(4, dtype=np.float32) # Terminalish
        
        # Push 10 identic samples
        for _ in range(10):
            buf.push(state, action, reward, next_state)
            
        # Initial Weights (Zero)
        # Input 4, Output 2 (Actions 0, 1)
        # Flattened size = 8
        weights = np.zeros(8, dtype=np.float32)
        
        # Train Loop
        errors = []
        for _ in range(20):
            weights, err = learner.train_weights_on_batch(weights, buf, batch_size=5, gamma=0.0)
            errors.append(err)
            
        print(f"   > Initial Error: {errors[0]:.4f}")
        print(f"   > Final Error: {errors[-1]:.4f}")
        
        self.assertLess(errors[-1], errors[0], "Error did not decrease!")
        # Check if weights for Action 0 increased (learned positive relation)
        # Weights indices for Action 0: 0, 1, 2, 3
        w_action_0_sum = np.sum(weights[0:4])
        self.assertGreater(w_action_0_sum, 0.5, "Weights for Action 0 did not adapt!")
        
        print("✅ JIT Learning Verified.")

if __name__ == '__main__':
    unittest.main()
