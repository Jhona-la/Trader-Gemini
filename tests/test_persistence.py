import unittest
import os
import shutil
import numpy as np
from strategies.technical import HybridScalpingStrategy
from core.genotype import Genotype

class MockDataProvider:
    def __init__(self):
        self.symbol_list = ['BTC/USDT']

class MockQueue:
    def put(self, item): pass

class TestPersistence(unittest.TestCase):
    def setUp(self):
        # Create temp dir for test genotypes
        if not os.path.exists("data/genotypes"):
            os.makedirs("data/genotypes")
            
    def test_save_on_stop(self):
        symbol = "TEST/PERSIST"
        
        # 1. Create Strategy with Genotype
        genotype = Genotype(symbol)
        genotype.init_brain(25, 4)
        
        # Modify weights to simulate learning
        # Set first weight to specific value
        genotype.genes['brain_weights'][0] = 999.99
        
        dp = MockDataProvider()
        queue = MockQueue()
        strategy = HybridScalpingStrategy(dp, queue, genotype=genotype)
        
        # 2. Stop Strategy (Should trigger save)
        strategy.stop()
        
        # 3. Verify File Exists
        filepath = f"data/genotypes/{symbol.replace('/','')}_gene.json"
        self.assertTrue(os.path.exists(filepath))
        
        # 4. Load and Verify
        loaded_genotype = Genotype.load(filepath)
        self.assertIsNotNone(loaded_genotype)
        self.assertEqual(loaded_genotype.genes['brain_weights'][0], 999.99)
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    unittest.main()
