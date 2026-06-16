#!/usr/bin/env python3
"""
===============================================================================
 GOD MODE SURROGATE: EVOLUTIVO, ADAPTATIVO E INTEGRAL (NEURAL EDITION)
===============================================================================
QUÉ: El oráculo supremo de optimización (Surrogate + Numba + Optuna).
     Ahora es 100% consciente de los horizontes de inversión (SCALPING vs SWING).
     Simula EL SISTEMA COMPLETO optimizando los 100 Pesos del Neural Bridge.
     Diseñado específicamente para portátiles de bajos recursos (vectorización total).
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import optuna
import argparse
import json
import gc

warnings.filterwarnings("ignore", category=UserWarning)
# Optimización para portátil sin GPU y pocos recursos:
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.surrogate_engine import ParetoSurrogateEnsemble
from core.simulation_multifidelity import run_neural_fidelity
from core.backtest_infra import fetch_multi_symbol_data

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("GodModeIntegral")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class HorizonConfig:
    """Configuraciones separadas para asegurar que Scalping y Swing no se pisen."""
    SCALPING = {
        'tp_pct': (0.001, 0.015),    # Micro ganancias rápidas
        'sl_pct': (0.001, 0.010),    # Riesgo ultrabajo (cuenta de 13 USD)
        'w_range': (-100.0, 100.0)   # Aumentado para romper softmax flatness
    }
    SWING = {
        'tp_pct': (0.020, 0.150),    # Movimientos macro
        'sl_pct': (0.015, 0.050),    # Holgura para volatilidad estructural
        'w_range': (-100.0, 100.0)   # Aumentado para romper softmax flatness
    }

def generate_random_params(num_samples: int, horizon: str) -> np.ndarray:
    """
    Genera 102 variables: [tp_pct, sl_pct, w_0, w_1, ..., w_99]
    """
    conf = HorizonConfig.SCALPING if horizon == 'scalping' else HorizonConfig.SWING
    tp = np.random.uniform(*conf['tp_pct'], num_samples)
    sl = np.random.uniform(*conf['sl_pct'], num_samples)
    
    # 100 Neural Weights (25 inputs x 4 outputs)
    weights = np.random.uniform(*conf['w_range'], (num_samples, 100))
    
    # Concatenar a forma (num_samples, 102)
    return np.column_stack((tp, sl, weights))

class GodModeSurrogate:
    def __init__(self, symbol: str, closes_arr: np.ndarray, horizon: str):
        self.symbol = symbol
        self.closes_arr = closes_arr
        self.horizon = horizon
        self.surrogate = ParetoSurrogateEnsemble(n_estimators=100) # Reducido para ahorrar RAM
        
    def _evaluate_param(self, p: np.ndarray, fidelity: str) -> dict:
        tp_pct = p[0]
        sl_pct = p[1]
        # Reformatear pesos a matriz (25, 4)
        weights_matrix = p[2:].reshape(25, 4)
        return run_neural_fidelity(self.closes_arr, tp_pct, sl_pct, weights_matrix, fidelity)

    def phase_explore(self):
        logger.info(f"[{self.horizon.upper()}] FASE 0: Recolectando 500 simulaciones NEURALES (F2) para entrenar el Cerebro Surrogate...")
        warmup_params = generate_random_params(500, self.horizon)
        
        y_pnl, y_dd, y_wr = [], [], []
        valid_warmup = 0
        for p in warmup_params:
            res = self._evaluate_param(p, 'F2')
            y_pnl.append(res['pnl'] if res['trades'] > 0 else 0.0)
            y_dd.append(res['max_dd'] if res['trades'] > 0 else 1.0)
            y_wr.append(res['win_rate'] if res['trades'] > 0 else 0.0)
            if res['trades'] > 0:
                valid_warmup += 1
                
        logger.info(f"[{self.horizon.upper()}] Simulaciones F2 con trades válidos: {valid_warmup}/500")
        self.surrogate.train(warmup_params, np.array(y_pnl), np.array(y_dd), np.array(y_wr))

    def phase_exploit(self):
        # 1 millón para pesos neuronales es muy manejable. (102 dims * 1M * 8 bytes = ~800 MB)
        VIRTUAL_SAMPLES = 1_000_000 
        logger.info(f"[{self.horizon.upper()}] FASE 1: Generando {VIRTUAL_SAMPLES:,} configs virtuales 102D (Neural Matrix)...")
        virtual_params = generate_random_params(VIRTUAL_SAMPLES, self.horizon)
        
        # Filtra las mejores usando Machine Learning instantáneo
        top_10k_params, _ = self.surrogate.filter_promising_configs(virtual_params, top_k=10000)
        logger.info(f"[{self.horizon.upper()}] ✅ Top 10,000 extraídas de la matriz virtual.")
        
        # Limpiamos para salvar RAM
        del virtual_params
        gc.collect()
        
        logger.info(f"[{self.horizon.upper()}] FASE 2: Filtrado F1 (Micro-Backtests NEURALES) de las Top 10,000...")
        f1_results = []
        t0 = time.time()
        for p in top_10k_params:
            res = self._evaluate_param(p, 'F1')
            f1_results.append(res['pnl'] if res['trades'] > 0 else -100.0)
        logger.info(f"[{self.horizon.upper()}] ✅ F1 completado en {time.time()-t0:.2f}s.")
        
        top_1k_indices = np.argsort(f1_results)[::-1][:1000]
        return top_10k_params[top_1k_indices]

    def phase_validate(self, top_1k_params):
        logger.info(f"[{self.horizon.upper()}] FASE 3: Validación F4 final usando Optuna TPE sobre pesos neuronales.")
        db_path = f'sqlite:///data/god_mode_{self.horizon}.db'
        study_name = f'GM_NEURAL_{self.horizon}_{self.symbol.replace("/", "")}'
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(),
            storage=optuna.storages.RDBStorage(url=db_path, engine_kwargs={"connect_args": {"timeout": 60}}),
            load_if_exists=True
        )
        
        # Inyectar el Top 50 del filtro cuántico
        conf = HorizonConfig.SCALPING if self.horizon == 'scalping' else HorizonConfig.SWING
        for p in top_1k_params[:50]:
            trial_dict = {'tp_pct': float(p[0]), 'sl_pct': float(p[1])}
            for i in range(100):
                trial_dict[f'w_{i}'] = float(p[2+i])
            study.enqueue_trial(trial_dict)
            
        def objective(trial):
            tp_pct = trial.suggest_float('tp_pct', *conf['tp_pct'])
            sl_pct = trial.suggest_float('sl_pct', *conf['sl_pct'])
            
            # Neural Weights
            weights = np.zeros(100, dtype=np.float64)
            for i in range(100):
                weights[i] = trial.suggest_float(f'w_{i}', *conf['w_range'])
                
            weights_matrix = weights.reshape(25, 4)
            res = run_neural_fidelity(self.closes_arr, tp_pct, sl_pct, weights_matrix, 'F4')
            
            score = res['pnl']
            
            # Penalizaciones Inteligentes y Suaves
            min_trades = 20 if self.horizon == 'scalping' else 5
            if res['trades'] < min_trades:
                score -= (min_trades - res['trades']) * 50.0  # Fuerte castigo por inactividad
                
            # Rigidez Extrema en el Drawdown por tener capital limitado (13 USD)
            max_allowed_dd = 0.02 if self.horizon == 'scalping' else 0.08
            if res['max_dd'] > max_allowed_dd:
                score -= (res['max_dd'] - max_allowed_dd) * 200.0
                
            # Exigencia CUÁNTICA de Win Rate para scalping
            if self.horizon == 'scalping':
                if res['win_rate'] < 98.0:
                    score -= (100.0 - res['win_rate']) * 100.0  # Penalización destructiva
                
            return score

        t_opt = time.time()
        study.optimize(objective, n_trials=100, n_jobs=1)
        
        logger.info(f"[{self.horizon.upper()}] 🏆 F4 Finalizado en {time.time()-t_opt:.2f}s!")
        logger.info(f"[{self.horizon.upper()}] 👑 MEJOR SCORE PnL: {study.best_value:.4f}")
        
        # Extraer parámetros para JSON
        best = study.best_trial.params
        final_genotype = {
            'symbol': self.symbol,
            'horizon': self.horizon,
            'genes': {
                'tp_pct': best['tp_pct'],
                'sl_pct': best['sl_pct'],
                'brain_weights': [best[f'w_{i}'] for i in range(100)]
            }
        }
        
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'genotypes'))
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f'god_mode_{self.horizon}_{self.symbol.replace("/", "")}.json')
        with open(file_path, 'w') as f:
            json.dump(final_genotype, f, indent=4)
        logger.info(f"[{self.horizon.upper()}] 💾 Genotipo Integral NEURAL Guardado en: {file_path}")
        
        # Limpieza activa de RAM
        del study
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help="Días de historial real de Binance")
    args = parser.parse_args()
    
    print("==================================================")
    print("🧠 GOD MODE SURROGATE: NEURAL INTEGRAL EDITION")
    print("==================================================")
    
    TARGET_COINS = ["BTC/USDT", "ETH/USDT"]
    logger.info(f"📡 Descargando datos masivos reales de Binance (últimos {args.days} días)...")
    all_data = fetch_multi_symbol_data(TARGET_COINS, args.days, max_workers=2)
    
    for symbol in TARGET_COINS:
        if symbol in all_data and not all_data[symbol].empty:
            closes = np.ascontiguousarray(all_data[symbol]['close'].values, dtype=np.float64)
            
            # Ejecutar de forma Integral y Separada: Scalping y Swing NO se pisan
            for horizon in ['scalping', 'swing']:
                print(f"\n" + "="*50)
                print(f"🚀 INICIANDO EVOLUCIÓN NEURAL PARA: {symbol} | MODO: {horizon.upper()}")
                print("="*50)
                god = GodModeSurrogate(symbol, closes, horizon)
                god.phase_explore()
                top_1k = god.phase_exploit()
                god.phase_validate(top_1k)
                
    print("\n✅ OPTIMIZACIÓN INTEGRAL NEURAL COMPLETADA PARA TODOS LOS ACTIVOS Y HORIZONTES.")

if __name__ == "__main__":
    main()
