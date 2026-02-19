import unittest
import numpy as np
from core.online_learning import OnlineLearner

class TestOnlineLearner(unittest.TestCase):
    def setUp(self):
        self.learner = OnlineLearner(learning_rate=0.1, clip_value=1.0) # High clip for math test
        
    def test_update_matrix_single_step(self):
        # Setup: 2 Inputs, 2 Outputs
        # Weights: Identity Matrix
        # [[1.0, 0.0],
        #  [0.0, 1.0]]
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # Input: [1.0, 1.0]
        inputs = np.array([1.0, 1.0])
        
        # We predicted Output 0.
        # Prediction: 1.0 * 1.0 + 0.0 * 1.0 = 1.0
        prediction = 1.0
        
        # Target: 0.0 (We wanted it lower)
        target = 0.0
        output_idx = 0
        
        # Expected Update:
        # Error = 0.0 - 1.0 = -1.0
        # Delta = 0.1 * -1.0 * [1.0, 1.0] = [-0.1, -0.1]
        # New Column 0 = [1.0, 0.0] + [-0.1, -0.1] = [0.9, -0.1]
        # Column 1 should be unchanged: [0.0, 1.0]
        
        new_weights = self.learner.update_matrix(weights, inputs, target, prediction, output_idx)
        
        expected_col0 = np.array([0.9, -0.1])
        expected_col1 = np.array([0.0, 1.0])
        
        np.testing.assert_array_almost_equal(new_weights[:, 0], expected_col0)
        np.testing.assert_array_almost_equal(new_weights[:, 1], expected_col1)

    def test_clipping(self):
        # Test large error to verify clipping
        learner_safe = OnlineLearner(learning_rate=0.1, clip_value=0.01)
        
        weights = np.zeros((2, 2))
        inputs = np.array([10.0, 10.0])
        target = 100.0
        prediction = 0.0
        output_idx = 1
        
        # Error = 100
        # Delta Raw = 0.1 * 100 * 10 = 100
        # Clipped Delta = 0.01
        
        new_weights = learner_safe.update_matrix(weights, inputs, target, prediction, output_idx)
        
        expected_val = 0.01
        self.assertAlmostEqual(new_weights[0, 1], expected_val)

if __name__ == '__main__':
    unittest.main()
