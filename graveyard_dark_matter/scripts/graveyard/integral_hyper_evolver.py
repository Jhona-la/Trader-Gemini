import os
import sys
import json
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from core.quantum_engine import QuantumEngine

logger = setup_logger("integral_hyper_evolver")
INTEGRAL_DNA_PATH = ".models/integral_quantum_dna.json"

class IntegralContinuousEvolver:
    """
    FASE 30: EVOLUCIONADOR INTEGRAL FULL-STACK
    Muta y evalúa TODO el sistema (Técnico + Machine Learning + Consenso + Gestión de Riesgo)
    al mismo tiempo en nanosegundos, para evitar que las áreas se contradigan en Producción.
    """
    def __init__(self, population_size=100):
        self.pop_size = population_size
        self.engine = QuantumEngine(capital=13.0, horizon="BOTH")
        self.engine.load_data(days=15)
        
        self.best_dna = self.load_dna()
        self.best_fitness = -999999
        
    def load_dna(self):
        if os.path.exists(INTEGRAL_DNA_PATH):
            try:
                with open(INTEGRAL_DNA_PATH, 'r') as f:
                    return json.load(f)
            except:
                from utils.error_handler import SystemIntegrityError
                raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
        
        # DNA SEMILLA FULL-STACK
        return {
            # 1. TECHNICAL STRATEGY (Scalping)
            'scalp_rsi_buy': 33,
            'scalp_rsi_sell': 71,
            'scalp_bb_std': 2.998,
            'scalp_tp_pct': 0.005,
            'scalp_sl_pct': 0.02,
            'scalp_ema_fast': 23,
            'scalp_ema_slow': 72,
            'scalp_leverage': 10.0,
            
            # 2. TECHNICAL STRATEGY (Swing)
            'swing_rsi_buy': 30,
            'swing_rsi_sell': 70,
            'swing_bb_std': 2.5,
            'swing_tp_pct': 0.02,
            'swing_sl_pct': 0.04,
            'swing_ema_fast': 50,
            'swing_ema_slow': 200,
            'swing_leverage': 5.0,
            
            # 3. MACHINE LEARNING THRESHOLDS
            'ml_threshold_bull': 0.55,
            'ml_threshold_bear': 0.55,
            
            # 4. CONSENSUS FILTER (Risk / Volatility Gate)
            'consensus_fee_mult': 2.0,
            
            # 5. OMNI-SCORE WEIGHTS (Fase 31 Binomio Perfecto)
            'w_technical': 1.0,
            'w_ml': 1.0,
            'w_phalanx': 0.5,
            'w_statarb': 0.5,
            'master_threshold': 1.5
        }
        
    def save_dna(self, dna):
        os.makedirs('.models', exist_ok=True)
        with open(INTEGRAL_DNA_PATH, 'w') as f:
            json.dump(dna, f, indent=4)
            
    def mutate(self, base_dna):
        mutated = base_dna.copy()
        mutation_rate = 0.3
        
        # --- TECHNICAL SCALP ---
        if random.random() < mutation_rate: mutated['scalp_rsi_buy'] = max(10, min(50, mutated['scalp_rsi_buy'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['scalp_rsi_sell'] = max(50, min(90, mutated['scalp_rsi_sell'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['scalp_bb_std'] = round(max(1.0, min(4.0, mutated['scalp_bb_std'] + random.uniform(-0.5, 0.5))), 3)
        if random.random() < mutation_rate: mutated['scalp_tp_pct'] = round(max(0.002, min(0.1, mutated['scalp_tp_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['scalp_sl_pct'] = round(max(0.01, min(0.1, mutated['scalp_sl_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['scalp_leverage'] = round(max(1.0, min(50.0, mutated['scalp_leverage'] + random.randint(-5, 5))), 1)
        
        # --- TECHNICAL SWING ---
        if random.random() < mutation_rate: mutated['swing_rsi_buy'] = max(10, min(50, mutated['swing_rsi_buy'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['swing_rsi_sell'] = max(50, min(90, mutated['swing_rsi_sell'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['swing_bb_std'] = round(max(1.0, min(4.0, mutated['swing_bb_std'] + random.uniform(-0.5, 0.5))), 3)
        if random.random() < mutation_rate: mutated['swing_tp_pct'] = round(max(0.01, min(0.2, mutated['swing_tp_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['swing_sl_pct'] = round(max(0.02, min(0.15, mutated['swing_sl_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['swing_leverage'] = round(max(1.0, min(30.0, mutated['swing_leverage'] + random.randint(-5, 5))), 1)
        
        # --- MACHINE LEARNING ---
        if random.random() < mutation_rate: mutated['ml_threshold_bull'] = round(max(0.40, min(0.80, mutated['ml_threshold_bull'] + random.uniform(-0.05, 0.05))), 3)
        if random.random() < mutation_rate: mutated['ml_threshold_bear'] = round(max(0.40, min(0.80, mutated['ml_threshold_bear'] + random.uniform(-0.05, 0.05))), 3)
        
        # --- CONSENSUS FILTER ---
        if random.random() < mutation_rate: mutated['consensus_fee_mult'] = round(max(0.5, min(5.0, mutated['consensus_fee_mult'] + random.uniform(-0.5, 0.5))), 2)
            
        # --- OMNI-SCORE WEIGHTS (Fase 31) ---
        if random.random() < mutation_rate: mutated['w_technical'] = round(max(0.0, min(3.0, mutated['w_technical'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate: mutated['w_ml'] = round(max(0.0, min(3.0, mutated['w_ml'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate: mutated['w_phalanx'] = round(max(0.0, min(3.0, mutated['w_phalanx'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate: mutated['w_statarb'] = round(max(0.0, min(3.0, mutated['w_statarb'] + random.uniform(-0.3, 0.3))), 2)
        if random.random() < mutation_rate: mutated['master_threshold'] = round(max(0.5, min(5.0, mutated['master_threshold'] + random.uniform(-0.5, 0.5))), 2)
        
        return mutated
        
    def calculate_fitness(self, results):
        pnl = results['pnl']
        wr = results['win_rate']
        trades = results['trades']
        
        if trades < 10 or pnl <= 0:
            return -9999.0
            
        # PnL Reina Suprema, penalidad dura al WinRate
        penalty = (100.0 - wr) * 10
        return pnl - penalty

    def run_perpetual_loop(self):
        logger.info("🧠 INICIANDO EVOLUCIÓN INTEGRAL FULL-STACK (ML + CONSENSO + TÉCNICO)")
        
        res = self.engine.run_vectorized_backtest(dna=self.best_dna)
        self.best_fitness = self.calculate_fitness(res)
        logger.info(f"📊 Base Fitness: {self.best_fitness:.2f} | PnL: ${res['pnl']:.2f} | WR: {res['win_rate']:.2f}% | Trades: {res['trades']}")
        
        cycle = 1
        while True:
            population = [self.best_dna]
            for _ in range(self.pop_size - 1):
                population.append(self.mutate(self.best_dna))
                
            best_cycle_fitness = -999999
            best_cycle_dna = None
            best_cycle_res = None
            
            for dna in population:
                # Filtrar liquidaciones puras
                if dna['scalp_sl_pct'] * dna['scalp_leverage'] >= 0.95: continue
                if dna['swing_sl_pct'] * dna['swing_leverage'] >= 0.95: continue
                    
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
                logger.info(f"🏆 [Ciclo {cycle}] NUEVO ADN INTEGRAL ENCONTRADO!")
                logger.info(f"📈 PnL: ${best_cycle_res['pnl']:.2f} | WR: {best_cycle_res['win_rate']:.2f}% | Trades: {best_cycle_res['trades']}")
                logger.info(f"🧠 ML Threshold: Bull {self.best_dna['ml_threshold_bull']} / Bear {self.best_dna['ml_threshold_bear']}")
                logger.info(f"🛡️ Consensus Fee Mult: {self.best_dna['consensus_fee_mult']}x")
            
            if cycle % 10 == 0:
                logger.info(f"🔄 Ciclo {cycle} completado. Rey actual retiene el trono (Fitness: {self.best_fitness:.2f})")
                
            cycle += 1

if __name__ == "__main__":
    evolver = IntegralContinuousEvolver(population_size=100)
    evolver.run_perpetual_loop()
