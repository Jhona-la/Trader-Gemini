#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 QUANTUM HYPER-EVOLVER (Vectorized Genetic Algorithm)
═══════════════════════════════════════════════════════════════════════════════
Ejecuta el QuantumEngine miles de veces por segundo para optimizar
el ADN de los bots (Estrategias) a través de una función de aptitud.
"""

import time
import argparse
import sys
import os
import random
import json
import numpy as np

# Project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.quantum_engine import QuantumEngine
from utils.logger import logger, stop_logger

class GeneticAlgorithm:
    def __init__(self, engine, population_size=1000, generations=50, mutation_rate=0.1):
        self.engine = engine
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        
    def _random_dna(self):
        return {
            'rsi_buy': random.randint(20, 45),
            'rsi_sell': random.randint(55, 80),
            'bb_std': round(random.uniform(1.5, 3.0), 1),
            'tp_pct': round(random.uniform(0.001, 0.05), 4),
            'sl_pct': round(random.uniform(0.001, 0.05), 4),
            'ema_fast': random.randint(5, 30),
            'ema_slow': random.randint(35, 100)
        }
        
    def init_population(self):
        self.population = [self._random_dna() for _ in range(self.pop_size)]
        
    def fitness(self, dna):
        results = self.engine.run_vectorized_backtest(dna=dna)
        # Priorities: Survival (WR=100%), Profit
        wr = results['win_rate']
        pnl = results['pnl']
        trades = results['trades']
        
        # Penalize low trades
        if trades < 5:
            return -9999.0
            
        # Hard constraint: Micro-account demands 100% win-rate for compounding
        if wr < 100.0:
            # Penalize linearly based on distance from 100%
            penalty = (100.0 - wr) * 10
            return pnl - penalty
            
        return pnl

    def _crossover(self, p1, p2):
        child = {}
        for key in p1.keys():
            if random.random() > 0.5:
                child[key] = p1[key]
            else:
                child[key] = p2[key]
        return child
        
    def _mutate(self, dna):
        mutated = dna.copy()
        for key in mutated.keys():
            if random.random() < self.mutation_rate:
                # Slight modification instead of total replacement
                if isinstance(mutated[key], int):
                    mutated[key] += random.choice([-2, -1, 1, 2])
                else:
                    mutated[key] += random.choice([-0.002, 0.002])
                    
        # Bounds checking
        mutated['rsi_buy'] = max(10, min(50, mutated['rsi_buy']))
        mutated['rsi_sell'] = max(50, min(90, mutated['rsi_sell']))
        mutated['tp_pct'] = max(0.001, mutated['tp_pct'])
        mutated['sl_pct'] = max(0.001, mutated['sl_pct'])
        if mutated['ema_fast'] >= mutated['ema_slow']:
            mutated['ema_fast'] = mutated['ema_slow'] - 5
            
        return mutated

    def evolve(self):
        logger.info(f"🧬 Iniciando Algoritmo Genético: Pop={self.pop_size}, Gens={self.generations}")
        self.init_population()
        
        best_overall = None
        best_fitness = -float('inf')
        best_results = None
        
        t0 = time.perf_counter()
        
        for gen in range(self.generations):
            scores = []
            for dna in self.population:
                fit = self.fitness(dna)
                scores.append((fit, dna))
                
            # Sort by fitness descending
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Elitism: Keep top 10%
            elite_count = int(self.pop_size * 0.1)
            new_population = [x[1] for x in scores[:elite_count]]
            
            # Track best
            if scores[0][0] > best_fitness:
                best_fitness = scores[0][0]
                best_overall = scores[0][1]
                best_results = self.engine.run_vectorized_backtest(dna=best_overall)
                logger.info(f"🏆 [Gen {gen}] Nuevo Rey Mutante! WR: {best_results['win_rate']:.2f}% | PnL: ${best_results['pnl']:.2f} | Trades: {best_results['trades']} | Fitness: {best_fitness:.2f}")
                
            # Crossover to fill rest
            while len(new_population) < self.pop_size:
                # Tournament selection
                p1 = random.choice(scores[:self.pop_size//2])[1]
                p2 = random.choice(scores[:self.pop_size//2])[1]
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)
                
            self.population = new_population
            
        t_total = time.perf_counter() - t0
        logger.info(f"✅ Evolución Completada en {t_total:.2f} segundos.")
        
        return best_overall, best_results

def parse_args():
    parser = argparse.ArgumentParser(description='Quantum Hyper-Evolver')
    parser.add_argument('--days', type=int, default=15, help='Days of historical data')
    parser.add_argument('--pop', type=int, default=500, help='Population size')
    parser.add_argument('--gens', type=int, default=30, help='Number of generations')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"🌌 QUANTUM HYPER-EVOLVER INICIALIZADO")
    
    engine = QuantumEngine(capital=13.0, horizon="SCALPING")
    logger.info("📡 Inyectando Parquets en RAM (Vector Load)...")
    engine.load_data(days=args.days)
    
    if len(engine.data) == 0:
        logger.error("❌ No data loaded.")
        sys.exit(1)
        
    ga = GeneticAlgorithm(engine, population_size=args.pop, generations=args.gens)
    best_dna, best_results = ga.evolve()
    
    logger.info("\n" + "="*50)
    logger.info("👑 ADN GANADOR (EL SANTO GRIAL)")
    logger.info("="*50)
    logger.info(json.dumps(best_dna, indent=4))
    logger.info("="*50)
    logger.info("📈 MÉTRICAS DEL ADN GANADOR:")
    logger.info(f"Win Rate : {best_results['win_rate']:.2f}%")
    logger.info(f"Trades   : {best_results['trades']}")
    logger.info(f"PnL Neto : ${best_results['pnl']:.2f}")
    logger.info("="*50)
    
    # Save DNA
    os.makedirs('.models', exist_ok=True)
    with open('.models/quantum_dna.json', 'w') as f:
        json.dump(best_dna, f, indent=4)
    logger.info("💾 ADN Guardado en .models/quantum_dna.json")
    
    stop_logger()

if __name__ == '__main__':
    main()
