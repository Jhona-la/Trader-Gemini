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
                from utils.error_handler import SystemIntegrityError
                raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
        # Default ADN de escalping
        return {
            'rsi_buy': 33,
            'rsi_sell': 71,
            'bb_std': 2.998,
            'tp_pct': 0.001,
            'sl_pct': 0.0484,
            'ema_fast': 23,
            'ema_slow': 72,
            'leverage': 10.0,
            'kelly_fraction': 0.30,
            'max_concurrent': 5,
            'scalp_w_ml': 1.0,
            'scalp_w_technical': 1.0,
            'scalp_ml_th_long': 0.55,
            'scalp_ml_th_short': 0.55,
            'scalp_master_threshold': 1.0,
            'swing_w_ml': 1.0,
            'swing_w_technical': 1.0,
            'swing_ml_th_long': 0.55,
            'swing_ml_th_short': 0.55,
            'swing_master_threshold': 1.0
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
            # Apalancamiento y Riesgo
            current_lev = mutated['leverage']
            mutated['leverage'] = round(max(1.0, min(50.0, current_lev + random.randint(-5, 5))), 1)
        if random.random() < mutation_rate:
            mutated['kelly_fraction'] = round(max(0.05, min(0.95, mutated['kelly_fraction'] + random.uniform(-0.1, 0.1))), 2)
        if random.random() < mutation_rate:
            mutated['max_concurrent'] = max(1, min(10, mutated['max_concurrent'] + random.randint(-2, 2)))
            
        # OmniScore & ML Thresh (Scalping)
        if random.random() < mutation_rate:
            mutated['scalp_w_ml'] = round(max(0.1, min(3.0, mutated['scalp_w_ml'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate:
            mutated['scalp_w_technical'] = round(max(0.1, min(3.0, mutated['scalp_w_technical'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate:
            mutated['scalp_master_threshold'] = round(max(0.5, min(3.0, mutated['scalp_master_threshold'] + random.uniform(-0.2, 0.2))), 2)
            
        # OmniScore & ML Thresh (Swing)
        if random.random() < mutation_rate:
            mutated['swing_w_ml'] = round(max(0.1, min(3.0, mutated['swing_w_ml'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate:
            mutated['swing_w_technical'] = round(max(0.1, min(3.0, mutated['swing_w_technical'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate:
            mutated['swing_master_threshold'] = round(max(0.5, min(3.0, mutated['swing_master_threshold'] + random.uniform(-0.2, 0.2))), 2)
            
        return mutated
        
    def calculate_fitness(self, results):
        pnl = results['pnl']
        wr = results['win_rate']
        trades = results['trades']
        
        if trades == 0:
            return -99999.0
            
        # Queremos maximizar PnL Exponencial y penalizar WR bajo
        # PnL es la metrica reina
        # Proveer un gradiente continuo para que el algoritmo pueda aprender a salir del PnL negativo.
        penalty = (100.0 - wr) * 5.0 # Penalidad más suave para permitir exploración
        fitness = pnl - penalty
        
        # Fomentar al menos 5 trades para significancia estadística
        if trades < 5:
            fitness -= (5 - trades) * 10.0
            
        return fitness

    def run_perpetual_loop(self):
        logger.info("🌌 INICIANDO EVOLUCIÓN PERPETUA (COMPOUNDING)")
        logger.info(f"🧬 ADN Base: {self.best_dna}")
        
        # Evaluar ADN base
        res = self.engine.run_vectorized_backtest(dna=self.best_dna)
        self.best_fitness = self.calculate_fitness(res)
        logger.info(f"📊 Base Fitness: {self.best_fitness:.2f} | PnL: ${res['pnl']:.2f} | WR: {res['win_rate']:.2f}% | Leverage: {self.best_dna['leverage']}x")
        
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
                if dna['sl_pct'] * dna['leverage'] >= 0.99:
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
