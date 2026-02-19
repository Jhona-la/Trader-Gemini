import unittest
import os
import shutil
from unittest.mock import MagicMock
from core.symbol_factory import SymbolFactory
from core.genotype import Genotype
from strategies.technical import HybridScalpingStrategy

class TestSymbolFactory(unittest.TestCase):
    def setUp(self):
        # Mock Engine components
        self.mock_engine = MagicMock()
        self.mock_engine.data_provider = MagicMock()
        self.mock_engine.events = MagicMock()
        self.mock_engine.data_provider.symbol_list = []
        
        # Initialize Factory
        self.factory = SymbolFactory(self.mock_engine)
        self.factory.genotype_dir = "tests/temp_genotypes" # Test dir
        if not os.path.exists(self.factory.genotype_dir):
            os.makedirs(self.factory.genotype_dir)

    def tearDown(self):
        if os.path.exists(self.factory.genotype_dir):
            shutil.rmtree(self.factory.genotype_dir)

    def test_spawn_organism_creates_genesis_genotype(self):
        symbol = "BTC/USDT"
        self.factory.spawn_organism(symbol)
        
        # Check if file created
        gene_path = os.path.join(self.factory.genotype_dir, "BTCUSDT_gene.json")
        self.assertTrue(os.path.exists(gene_path))
        
        # Check strategy registration
        self.mock_engine.register_strategy.assert_called_once()
        strategy = self.mock_engine.register_strategy.call_args[0][0]
        
        self.assertIsInstance(strategy, HybridScalpingStrategy)
        self.assertEqual(strategy.symbol, symbol)
        self.assertIsNotNone(strategy.genotype)
        self.assertEqual(strategy.genotype.genes['tp_pct'], 0.015) # Default Genesis value

    def test_load_existing_genotype(self):
        symbol = "ETH/USDT"
        # Create a custom genotype
        custom_gene = Genotype(symbol=symbol)
        custom_gene.genes['tp_pct'] = 0.05 # Custom value
        custom_gene.save(os.path.join(self.factory.genotype_dir, "ETHUSDT_gene.json"))
        
        self.factory.spawn_organism(symbol)
        
        strategy = self.mock_engine.register_strategy.call_args[0][0]
        self.assertEqual(strategy.genotype.genes['tp_pct'], 0.05)

    def test_kill_organism(self):
        symbol = "SOL/USDT"
        self.factory.spawn_organism(symbol)
        self.assertIn(symbol, self.factory.active_organisms)
        
        self.factory.kill_organism(symbol)
        self.assertNotIn(symbol, self.factory.active_organisms)
        self.mock_engine.unregister_strategy.assert_called_with(symbol)

if __name__ == '__main__':
    unittest.main()
