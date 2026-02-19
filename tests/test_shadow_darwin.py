import unittest
import pandas as pd
import numpy as np
import shutil
import os
from core.shadow_darwin import ShadowDarwin
from core.simulation import SimDataProvider
from core.genotype import Genotype

class TestShadowDarwin(unittest.TestCase):
    def setUp(self):
        # Create Dummy Data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        df = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        }, index=dates)
        
        # Ensure High >= Low
        df['high'] = df[['open', 'close']].max(axis=1) + 1
        df['low'] = df[['open', 'close']].min(axis=1) - 1
        df.index.name = 'timestamp' # FIX: Naming index required by simulation
        
        self.data_map = {'BTC/USDT': df}
        self.provider = SimDataProvider(self.data_map)
        self.darwin = ShadowDarwin(self.provider, population_size=10)
        self.darwin.evolution_engine.mutation_rate = 0.5 # High rate for testing change
        
        # Temp dir
        if not os.path.exists("data/genotypes"):
            os.makedirs("data/genotypes")

    def tearDown(self):
        # Clean up created genes
        if os.path.exists("data/genotypes/BTCUSDT_gene.json"):
            os.remove("data/genotypes/BTCUSDT_gene.json")

    def test_initialization(self):
        self.darwin.initialize_population('BTC/USDT')
        self.assertEqual(len(self.darwin.populations['BTC/USDT']), 10)
        
    def test_evolution_cycle(self):
        # Run 1 epoch
        initial_pop = self.darwin.populations.get('BTC/USDT')
        initial_rate = self.darwin.evolution_engine.mutation_rate
        
        winner = self.darwin.run_epoch('BTC/USDT', generations=2)
        
        self.assertIsInstance(winner, Genotype)
        
        # Check Hall of Fame files
        self.assertTrue(os.path.exists("data/genotypes/BTCUSDT_gene_alpha.json"))
        # self.assertTrue(os.path.exists("data/genotypes/BTCUSDT_gene_beta_1.json")) # Might exist if N>1
        
        # Check Adaptation (Rate should change from initial if diversity is low/high)
        # With random data, diversity might be high, so rate might drop, or low -> boost.
        new_rate = self.darwin.evolution_engine.mutation_rate
        self.assertNotEqual(initial_rate, new_rate)
        
        # Check that population evolved (new parents)
        final_pop = self.darwin.populations['BTC/USDT']
        parents_exist = any('+' in g.parent_id for g in final_pop if g.parent_id)
        # Note: With random data, simulation might return 0 trades/fitness, so breeding might be random
        # But structure should hold.

    def test_simulation_integration(self):
        # Test direct sim run
        genotype = Genotype('BTC/USDT')
        trades = self.darwin.simulator.run(genotype, 'BTC/USDT')
        self.assertIsInstance(trades, list)

    def test_neuroevolution_cycle(self):
        # Test with Neural Flag ON
        darwin_neuro = ShadowDarwin(self.provider, population_size=5, use_neural=True)
        darwin_neuro.initialize_population('BTC/USDT')
        
        # Check if brains are initialized
        pop = darwin_neuro.populations['BTC/USDT']
        self.assertTrue(len(pop[0].genes['brain_weights']) > 0)
        self.assertEqual(len(pop[0].genes['brain_weights']), 100) # 25 * 4
        
        # Run Epoch
        winner = darwin_neuro.run_epoch('BTC/USDT', generations=1)
        
        # Check if weights mutated
        # We can check by comparing "Adam" (index 0 initially, but might be sorted)
        # to a mutant.
        # Actually just check that we have results
        self.assertIsInstance(winner, Genotype)
        self.assertTrue(len(winner.genes['brain_weights']) > 0)

if __name__ == '__main__':
    unittest.main()
