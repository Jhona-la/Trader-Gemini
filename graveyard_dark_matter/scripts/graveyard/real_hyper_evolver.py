import os
import sys
import json
import time
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from core.quantum_engine import QuantumEngine

logger = setup_logger("real_hyper_evolver")
REAL_DNA_PATH = ".models/real_quantum_dna.json"

class RealContinuousEvolver:
    def __init__(self, population_size=100):
        self.pop_size = population_size
        self.engine = QuantumEngine(capital=13.0, horizon="BOTH")
        self.engine.load_data(days=15)
        
        self.best_dna = self.load_dna()
        self.best_fitness = -999999
        
    def load_dna(self):
        if os.path.exists(REAL_DNA_PATH):
            try:
                with open(REAL_DNA_PATH, 'r') as f:
                    return json.load(f)
            except:
                from utils.error_handler import SystemIntegrityError
                raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
        
        return {
            'scalp_rsi_buy': 33,
            'scalp_rsi_sell': 71,
            'scalp_bb_std': 2.998,
            'scalp_tp_pct': 0.005,
            'scalp_sl_pct': 0.02,
            'scalp_ema_fast': 23,
            'scalp_ema_slow': 72,
            'scalp_leverage': 10.0,
            'scalp_ml_th_long': 0.65,
            'scalp_ml_th_short': 0.35,
            'scalp_master_threshold': 1.0,
            'scalp_w_ml': 1.0,
            'scalp_w_technical': 1.0,
            'scalp_w_phalanx': 0.5,
            'scalp_w_statarb': 0.5,
            
            'swing_rsi_buy': 30,
            'swing_rsi_sell': 70,
            'swing_bb_std': 2.5,
            'swing_tp_pct': 0.02,
            'swing_sl_pct': 0.04,
            'swing_ema_fast': 50,
            'swing_ema_slow': 200,
            'swing_leverage': 5.0,
            'swing_ml_th_long': 0.60,
            'swing_ml_th_short': 0.40,
            'swing_master_threshold': 1.0,
            'swing_w_ml': 1.0,
            'swing_w_technical': 1.0,
            'swing_w_phalanx': 0.5,
            'swing_w_statarb': 0.5
        }
        
    def save_dna(self, dna):
        os.makedirs('.models', exist_ok=True)
        with open(REAL_DNA_PATH, 'w') as f:
            json.dump(dna, f, indent=4)
            
    def mutate(self, base_dna):
        mutated = base_dna.copy()
        mutation_rate = 0.3
        
        # Scalping Mutations
        if random.random() < mutation_rate: mutated['scalp_rsi_buy'] = max(10, min(50, mutated['scalp_rsi_buy'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['scalp_rsi_sell'] = max(50, min(90, mutated['scalp_rsi_sell'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['scalp_bb_std'] = round(max(1.0, min(4.0, mutated['scalp_bb_std'] + random.uniform(-0.5, 0.5))), 3)
        if random.random() < mutation_rate: mutated['scalp_tp_pct'] = round(max(0.002, min(0.1, mutated['scalp_tp_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['scalp_sl_pct'] = round(max(0.01, min(0.1, mutated['scalp_sl_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['scalp_leverage'] = round(max(1.0, min(50.0, mutated['scalp_leverage'] + random.randint(-5, 5))), 1)
        if random.random() < mutation_rate: mutated['scalp_ml_th_long'] = round(max(0.50, min(0.85, mutated['scalp_ml_th_long'] + random.uniform(-0.05, 0.05))), 3)
        if random.random() < mutation_rate: mutated['scalp_ml_th_short'] = round(max(0.15, min(0.50, mutated['scalp_ml_th_short'] + random.uniform(-0.05, 0.05))), 3)
        if random.random() < mutation_rate: mutated['scalp_master_threshold'] = round(max(0.1, min(3.0, mutated['scalp_master_threshold'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['scalp_w_ml'] = round(max(0.0, min(2.0, mutated['scalp_w_ml'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['scalp_w_technical'] = round(max(0.0, min(2.0, mutated['scalp_w_technical'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['scalp_w_phalanx'] = round(max(0.0, min(2.0, mutated['scalp_w_phalanx'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['scalp_w_statarb'] = round(max(0.0, min(2.0, mutated['scalp_w_statarb'] + random.uniform(-0.2, 0.2))), 3)

        
        # Swing Mutations
        if random.random() < mutation_rate: mutated['swing_rsi_buy'] = max(10, min(50, mutated['swing_rsi_buy'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['swing_rsi_sell'] = max(50, min(90, mutated['swing_rsi_sell'] + random.randint(-5, 5)))
        if random.random() < mutation_rate: mutated['swing_bb_std'] = round(max(1.0, min(4.0, mutated['swing_bb_std'] + random.uniform(-0.5, 0.5))), 3)
        if random.random() < mutation_rate: mutated['swing_tp_pct'] = round(max(0.01, min(0.2, mutated['swing_tp_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['swing_sl_pct'] = round(max(0.02, min(0.15, mutated['swing_sl_pct'] * random.uniform(0.8, 1.2))), 4)
        if random.random() < mutation_rate: mutated['swing_leverage'] = round(max(1.0, min(30.0, mutated['swing_leverage'] + random.randint(-5, 5))), 1)
        if random.random() < mutation_rate: mutated['swing_ml_th_long'] = round(max(0.50, min(0.85, mutated['swing_ml_th_long'] + random.uniform(-0.05, 0.05))), 3)
        if random.random() < mutation_rate: mutated['swing_ml_th_short'] = round(max(0.15, min(0.50, mutated['swing_ml_th_short'] + random.uniform(-0.05, 0.05))), 3)
        if random.random() < mutation_rate: mutated['swing_master_threshold'] = round(max(0.1, min(3.0, mutated['swing_master_threshold'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['swing_w_ml'] = round(max(0.0, min(2.0, mutated['swing_w_ml'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['swing_w_technical'] = round(max(0.0, min(2.0, mutated['swing_w_technical'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['swing_w_phalanx'] = round(max(0.0, min(2.0, mutated['swing_w_phalanx'] + random.uniform(-0.2, 0.2))), 3)
        if random.random() < mutation_rate: mutated['swing_w_statarb'] = round(max(0.0, min(2.0, mutated['swing_w_statarb'] + random.uniform(-0.2, 0.2))), 3)

            
        return mutated
        
    def calculate_fitness(self, results):
        pnl = results['pnl']
        wr = results['win_rate']
        trades = results['trades']
        max_dd = results['max_drawdown']
        
        if trades < 15 or pnl <= 0:
            return -999999.0
            
        # FASE 67: Kelly Crecimiento Exponencial (100% cada 3 días)
        # 15 días de data = 5 ciclos de 3 días = capital_inicial * (2^5) = 32x (3100% ROI)
        
        capital_final = self.engine.capital + pnl
        multiplier = capital_final / self.engine.capital
        
        # 1. Tasa de Crecimiento (Evaluación Logarítmica)
        if multiplier > 0:
            growth_score = (multiplier ** (3.0 / 15.0)) * 100.0  # Normalized 3-day growth
        else:
            growth_score = -100.0
            
        # 2. Penalización por pérdida inaceptable (Drawdown / Ruina)
        # Queremos proteger el capital. Max DD > 10% es fatal.
        dd_penalty = 0
        if max_dd > 0.10: 
            dd_penalty = (max_dd * 100000)
            
        # 3. Requisito de Win Rate: Solo nos importa que el EV (Expected Value) total crezca.
        # Quitamos la penalización matemática estricta de WR y la reemplazamos con la del Criterio Kelly puro.
        kelly_fraction = 0.0
        if trades > 0:
            win_prob = wr / 100.0
            loss_prob = 1.0 - win_prob
            # Asumimos promedios de TP y SL desde los resultados para Kelly real (usamos aproximación rápida)
            avg_win = pnl / trades if pnl > 0 else 0.01 
            avg_loss = abs(pnl) / trades if pnl < 0 else 0.01
            
            # Simplified Kelly
            if avg_loss > 0:
                kelly_fraction = win_prob - (loss_prob / (avg_win / avg_loss))
                
        kelly_penalty = 0
        if kelly_fraction < 0.0:
            kelly_penalty = abs(kelly_fraction) * 10000 # Negative Kelly means guaranteed ruin over time
            
        # Fitness final: Recompensa crecimiento geométrico + Castiga DD y Kelly negativo
        fitness = growth_score - dd_penalty - kelly_penalty
        
        return fitness

    def run_perpetual_loop(self):
        logger.info("🌍 INICIANDO EVOLUCIÓN REALISTA (MULTI-HORIZONTE SCALP+SWING)")
        
        res = self.engine.run_vectorized_backtest(dna=self.best_dna)
        self.best_fitness = self.calculate_fitness(res)
        logger.info(f"📊 Base Fitness: {self.best_fitness:.2f} | PnL: ${res['pnl']:.2f} | WR: {res['win_rate']:.2f}%")
        
        cycle = 1
        while True:
            population = [self.best_dna]
            for _ in range(self.pop_size - 1):
                population.append(self.mutate(self.best_dna))
                
            best_cycle_fitness = -999999
            best_cycle_dna = None
            best_cycle_res = None
            
            for dna in population:
                # Evitar liquidación garantizada matemática
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
                logger.info(f"🏆 [Ciclo {cycle}] NUEVO REY MULTI-HORIZONTE!")
                logger.info(f"📈 PnL: ${best_cycle_res['pnl']:.2f} | WR: {best_cycle_res['win_rate']:.2f}% | Trades: {best_cycle_res['trades']}")
                logger.info(f"🧬 Scalp: Lev {self.best_dna['scalp_leverage']}x, TP {self.best_dna['scalp_tp_pct']}, SL {self.best_dna['scalp_sl_pct']}")
                logger.info(f"🧬 Swing: Lev {self.best_dna['swing_leverage']}x, TP {self.best_dna['swing_tp_pct']}, SL {self.best_dna['swing_sl_pct']}")
            
            if cycle % 10 == 0:
                logger.info(f"🔄 Ciclo {cycle} completado. Rey actual retiene el trono (Fitness: {self.best_fitness:.2f})")
                
            cycle += 1

if __name__ == "__main__":
    evolver = RealContinuousEvolver(population_size=100)
    evolver.run_perpetual_loop()
