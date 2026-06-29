import unittest
from core.evolution import EvolutionEngine, FitnessCalculator, TradeResult
from core.genotype import Genotype

class TestEvolutionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EvolutionEngine(mutation_rate=1.0, mutation_strength=0.1)
        
    def test_fitness_calculation(self):
        # Good trades
        trades = [
            TradeResult(pnl_pct=0.05, duration_seconds=60, is_win=True),
            TradeResult(pnl_pct=0.02, duration_seconds=60, is_win=True),
            TradeResult(pnl_pct=-0.01, duration_seconds=60, is_win=False)
        ]
        score = FitnessCalculator.calculate_fitness(trades)
        self.assertGreater(score, 0)
        
        # Bad trades
        bad_trades = [
            TradeResult(pnl_pct=-0.05, duration_seconds=60, is_win=False),
            TradeResult(pnl_pct=-0.02, duration_seconds=60, is_win=False)
        ]
        bad_score = FitnessCalculator.calculate_fitness(bad_trades)
        self.assertEqual(bad_score, 0) # Should be penalized to 0

    def test_selection_tournament(self):
        pop = []
        for i in range(5):
            g = Genotype(symbol="BTC/USDT")
            g.fitness_score = i * 10 
            pop.append(g)
            
        # Should pick the one with fitness 40
        winner = self.engine.select_parents_tournament(pop, tournament_size=3)
        self.assertGreaterEqual(winner.fitness_score, 20) # Probabilistic but high chance

    def test_crossover(self):
        parent_a = Genotype(symbol="TEST")
        parent_a.genes['tp_pct'] = 0.01
        
        parent_b = Genotype(symbol="TEST")
        parent_b.genes['tp_pct'] = 0.05
        
        child = Genotype(symbol="TEST")
        
        # Test 10 times to ensure mixing happens
        mixed = False
        for _ in range(10):
            child = self.engine.crossover(parent_a, parent_b, child)
            if child.genes['tp_pct'] in [0.01, 0.05]:
                mixed = True
        
        self.assertTrue(mixed)
        self.assertIn('+', child.parent_id)

    def test_mutation(self):
        genotype = Genotype(symbol="TEST")
        original_tp = genotype.genes['tp_pct']
        
        # Mutation rate 1.0 -> Always mutate
        mutated = self.engine.mutate(genotype)
        
        self.assertNotEqual(mutated.genes['tp_pct'], original_tp)

if __name__ == '__main__':
    unittest.main()
