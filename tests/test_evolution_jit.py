import unittest
import numpy as np
from core.genotype import Genotype
from core.evolution import EvolutionEngine

class TestEvolutionJIT(unittest.TestCase):
    def setUp(self):
        self.engine = EvolutionEngine(mutation_rate=0.5, mutation_strength=0.2)
        self.symbol = "BTC/USDT"
        
    def test_vectorized_generation(self):
        """Verify the full JIT generation loop"""
        print("\n🧬 Testing JIT Evolution Pipeline...")
        
        # 1. Create Population
        pop_size = 20
        population = [Genotype(symbol=self.symbol) for _ in range(pop_size)]
        
        # Init Brains
        for ind in population:
            ind.init_brain(10, 2) # small brain
            # Assign fake fitness
            ind.fitness_score = np.random.random() * 10.0
            
        # Sort initial
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        top_fitness_initial = population[0].fitness_score
        print(f"   > Initial Top Fitness: {top_fitness_initial:.4f}")
        
        # 2. Evolve
        elite_count = 2
        new_pop = self.engine.evolve_generation(population, elite_count)
        
        # 3. Verifications
        self.assertEqual(len(new_pop), pop_size, "Population size must be constant")
        
        # Check Elitism
        self.assertEqual(new_pop[0].fitness_score, top_fitness_initial, "Elitism failed: Top 1 lost")
        self.assertEqual(new_pop[1].fitness_score, population[1].fitness_score, "Elitism failed: Top 2 lost")
        
        # Check Mutation (New individuals should have zero fitness initially or at least distinct genes)
        # In our logic, we didn't reset fitness, but new objects start with 0.0 in Genotype __init__
        self.assertEqual(new_pop[-1].fitness_score, 0.0, "New child should have 0 fitness until evaluated")
        
        # Check Gene Drift
        old_val = population[-1].genes['rsi_period']
        new_val = new_pop[-1].genes['rsi_period']
        # Note: Might be same if mutation missed, but with rate 0.5 it's unlikely for ALL genes
        # Let's check array equality of brain weights
        
        old_brain = np.array(population[-1].genes['brain_weights'])
        new_brain = np.array(new_pop[-1].genes['brain_weights'])
        
        # They should be different
        self.assertFalse(np.array_equal(old_brain, new_brain), "Brain weights did not mutate!")
        
        print("✅ JIT Evolution Pipeline Verified.")

if __name__ == '__main__':
    unittest.main()
