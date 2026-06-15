import os
import sys
import json
import time
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from core.quantum_engine import QuantumEngine

logger = setup_logger("continuous_evolver")
DNA_PATH = ".models/quantum_dna.json"

class ContinuousEvolver:
    def __init__(self, population_size=100, mutations_per_cycle=5):
        self.pop_size = population_size
        self.mutations_per_cycle = mutations_per_cycle
        self.engine = QuantumEngine(capital=13.0)
        self.engine.load_data(days=15)
        
        # Cargar ADN base si existe
        self.best_dna = self.load_dna()
        self.best_fitness = -999999
        
    def load_dna(self):
        if os.path.exists(DNA_PATH):
            try:
                with open(DNA_PATH, 'r') as f:
                    return json.load(f)
            except:
                pass
        # Default ADN de escalping
        return {
            'rsi_buy': 33,
            'rsi_sell': 71,
            'bb_std': 2.998,
            'tp_pct': 0.001,
            'sl_pct': 0.0484,
            'ema_fast': 23,
            'ema_slow': 72,
            'leverage': 10.0 # Leverage base
        }
        
    def save_dna(self, dna):
        os.makedirs('.models', exist_ok=True)
        with open(DNA_PATH, 'w') as f:
            json.dump(dna, f, indent=4)
            
    def mutate(self, base_dna):
        mutated = base_dna.copy()
        mutation_rate = 0.3 # 30% chance to mutate a gene
        
        if random.random() < mutation_rate:
            mutated['rsi_buy'] = max(10, min(50, mutated['rsi_buy'] + random.randint(-5, 5)))
        if random.random() < mutation_rate:
            mutated['rsi_sell'] = max(50, min(90, mutated['rsi_sell'] + random.randint(-5, 5)))
        if random.random() < mutation_rate:
            mutated['bb_std'] = round(max(1.0, min(4.0, mutated['bb_std'] + random.uniform(-0.5, 0.5))), 3)
        if random.random() < mutation_rate:
            mutated['tp_pct'] = round(max(0.0005, min(0.1, mutated['tp_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate:
            mutated['sl_pct'] = round(max(0.005, min(0.2, mutated['sl_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate:
            mutated['ema_fast'] = max(5, min(40, mutated['ema_fast'] + random.randint(-3, 3)))
        if random.random() < mutation_rate:
            mutated['ema_slow'] = max(40, min(100, mutated['ema_slow'] + random.randint(-5, 5)))
        if random.random() < mutation_rate:
            # Apalancamiento agresivo
            current_lev = mutated.get('leverage', 1.0)
            mutated['leverage'] = round(max(1.0, min(50.0, current_lev + random.randint(-5, 5))), 1)
            
        return mutated
        
    def calculate_fitness(self, results):
        pnl = results['pnl']
        wr = results['win_rate']
        trades = results['trades']
        
        if trades < 5 or pnl <= 0:
            return -9999.0
            
        # Queremos maximizar PnL Exponencial y penalizar WR bajo
        # PnL es la metrica reina
        penalty = (100.0 - wr) * 50 # Severa penalidad por perder WR
        fitness = pnl - penalty
        return fitness

    def run_perpetual_loop(self):
        logger.info("🌌 INICIANDO EVOLUCIÓN PERPETUA (COMPOUNDING)")
        logger.info(f"🧬 ADN Base: {self.best_dna}")
        
        # Evaluar ADN base
        res = self.engine.run_vectorized_backtest(dna=self.best_dna)
        self.best_fitness = self.calculate_fitness(res)
        logger.info(f"📊 Base Fitness: {self.best_fitness:.2f} | PnL: ${res['pnl']:.2f} | WR: {res['win_rate']:.2f}% | Leverage: {self.best_dna.get('leverage', 1.0)}x")
        
        cycle = 1
        while True:
            population = [self.best_dna]
            for _ in range(self.pop_size - 1):
                population.append(self.mutate(self.best_dna))
                
            best_cycle_fitness = -999999
            best_cycle_dna = None
            best_cycle_res = None
            
            for dna in population:
                # Si el SL con apalancamiento supera 100%, es liquidación segura, descartar
                if dna['sl_pct'] * dna.get('leverage', 1.0) >= 0.99:
                    continue
                    
                res = self.engine.run_vectorized_backtest(dna=dna)
                fit = self.calculate_fitness(res)
                
                if fit > best_cycle_fitness:
                    best_cycle_fitness = fit
                    best_cycle_dna = dna
                    best_cycle_res = res
                    
            if best_cycle_fitness > self.best_fitness:
                self.best_fitness = best_cycle_fitness
                self.best_dna = best_cycle_dna
                self.save_dna(self.best_dna)
                logger.info(f"🏆 [Ciclo {cycle}] NUEVO REY SUPREMO!")
                logger.info(f"📈 PnL: ${best_cycle_res['pnl']:.2f} | WR: {best_cycle_res['win_rate']:.2f}% | Trades: {best_cycle_res['trades']}")
                logger.info(f"🧬 ADN: Leverage {self.best_dna['leverage']}x | TP: {self.best_dna['tp_pct']} | SL: {self.best_dna['sl_pct']}")
            
            # Cada 10 ciclos informamos
            if cycle % 10 == 0:
                logger.info(f"🔄 Ciclo {cycle} completado. Rey actual retiene el trono (Fitness: {self.best_fitness:.2f})")
                
            cycle += 1

if __name__ == "__main__":
    evolver = ContinuousEvolver(population_size=100)
    evolver.run_perpetual_loop()
