import unittest
import numpy as np
from core.neural_bridge import NeuralBridge
from core.genotype import Genotype

class TestNeuralBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = NeuralBridge(observation_window=5)
        self.genotype = Genotype(symbol="BTC/USDT")
        
    def test_tensor_shape(self):
        # Create dummy market data (structured array style)
        dtype = [('close', 'f4'), ('volume', 'f4')]
        data = np.zeros(10, dtype=dtype)
        data['close'] = np.random.rand(10) * 100
        data['volume'] = np.random.rand(10) * 1000
        
        portfolio_state = {'quantity': 0, 'pnl_pct': 0.0, 'duration': 0}
        
        tensor = self.bridge.get_state_tensor(data, portfolio_state, self.genotype)
        
        # Expected dim: (4 * 5) + 3 + 2 = 20 + 3 + 2 = 25
        expected_dim = (4 * 5) + 3 + 2
        
        self.assertEqual(tensor.shape[0], expected_dim)
        self.assertFalse(np.isnan(tensor).any())

    def test_insufficient_data(self):
        # Data valid but too short
        dtype = [('close', 'f4'), ('volume', 'f4')]
        data = np.zeros(2, dtype=dtype) # Only 2 bars
        
        tensor = self.bridge.get_state_tensor(data, {}, self.genotype)
        
        # Should return tensor with zeroed market data but valid genotype data
        expected_dim = (4 * 5) + 3 + 2
        self.assertEqual(tensor.shape[0], expected_dim)
        
        # Market data part (first 20) should be 0
        self.assertEqual(np.sum(tensor[:20]), 0)
        
        
        # Genotype part (last 2) should NOT be 0 (default genes)
        self.assertNotEqual(np.sum(tensor[-2:]), 0)

    def test_normalization(self):
        # Test genotype normalization
        self.genotype.genes['tp_pct'] = 0.05 # 5%
        
        tensor = self.bridge.get_state_tensor(None, {}, self.genotype)
        
        # Last element should be tp_norm
        # 0.05 * 10 = 0.5
        self.assertAlmostEqual(tensor[-1], 0.5)

    def test_decode_action(self):
        # 1. Test BUY (Index 1)
        probs = np.array([0.1, 0.8, 0.05, 0.05])
        signal, conf = self.bridge.decode_action(probs)
        
        # We need to import SignalType in test or check string repo of enum
        # Ideally import SignalType in test file
        # For now, let's assume SignalType.LONG is what we get
        # We will need to add import to test file
        self.assertEqual(signal.name, "LONG")
        self.assertAlmostEqual(conf, 0.8)
        
        # 2. Test HOLD (Index 0)
        probs = np.array([0.9, 0.05, 0.05, 0.0])
        signal, conf = self.bridge.decode_action(probs)
        self.assertIsNone(signal)
        
        # 3. Test CLOSE (Index 3)
        probs = np.array([0.1, 0.1, 0.1, 0.7])
        signal, conf = self.bridge.decode_action(probs)
        self.assertEqual(signal, "CLOSE")


if __name__ == '__main__':
    unittest.main()
