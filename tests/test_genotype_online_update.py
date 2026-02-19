import unittest
import numpy as np
from core.genotype import Genotype
from core.online_learning import OnlineLearner

class TestGenotypeOnlineUpdate(unittest.TestCase):
    def test_genotype_brain_update(self):
        # 1. Initialize Genotype and Brain
        symbol = "BTC/USDT"
        genotype = Genotype(symbol)
        input_dim = 25
        output_dim = 4
        genotype.init_brain(input_dim, output_dim)
        
        original_weights = np.array(genotype.genes['brain_weights'])
        original_matrix = original_weights.reshape(input_dim, output_dim)
        
        # 2. Setup Learner
        learner = OnlineLearner(learning_rate=0.1, clip_value=1.0)
        
        # 3. Simulate an Interaction
        inputs = np.ones(input_dim)
        target = 1.0
        prediction = 0.0 # Error = 1.0
        output_idx = 0 # Action 0 was taken
        
        # 4. Perform Update
        new_matrix = learner.update_matrix(original_matrix, inputs, target, prediction, output_idx)
        
        # 5. Save back to Genotype
        genotype.genes['brain_weights'] = new_matrix.flatten().tolist()
        
        # 6. Verify Change
        new_weights = np.array(genotype.genes['brain_weights'])
        
        # Check that Column 0 changed
        # Delta = 0.1 * 1.0 * 1.0 = 0.1
        # Original was likely Random small numbers.
        # Let's check the difference
        diff = new_weights.reshape(input_dim, output_dim) - original_matrix
        
        # Column 0 should have changed by +0.1
        np.testing.assert_array_almost_equal(diff[:, 0], np.full(input_dim, 0.1))
        
        # Column 1 should be 0 change
        np.testing.assert_array_almost_equal(diff[:, 1], np.zeros(input_dim))
        
        # 7. Verify JSON Serializability (Implicit in .tolist())
        self.assertIsInstance(genotype.genes['brain_weights'], list)

if __name__ == '__main__':
    unittest.main()
