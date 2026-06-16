#!/usr/bin/env python3
"""
===============================================================================
 QUANTUM SURROGATE EVOLVER (Multi-Fidelity BOHB)
===============================================================================
QUÉ: El "Santo Grial" del Optimizador Híbrido. 
     Combina LightGBM (Surrogate), Numba JIT (Vectorización), y 
     Multi-Fidelidad (Hyperband) para explorar 10 Millones de simulaciones 
     virtuales y miles reales en tiempo récord.
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import optuna
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["OMP_NUM_THREADS"] = "6"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.surrogate_engine import ParetoSurrogateEnsemble
from core.simulation_multifidelity import run_fidelity
from core.backtest_infra import fetch_multi_symbol_data

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("QuantumSurrogate")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generate_random_params(num_samples: int) -> np.ndarray:
    """
    Genera `num_samples` configuraciones aleatorias.
    Columnas: [tp_pct, sl_pct, window, fast_window, trend_conf]
    """
    tp = np.random.uniform(0.005, 0.05, num_samples)
    sl = np.random.uniform(0.005, 0.03, num_samples)
    w = np.random.randint(10, 50, num_samples)
    fw = np.random.randint(3, 20, num_samples)
    tc = np.random.uniform(0.0001, 0.005, num_samples)
    
    return np.column_stack((tp, sl, w, fw, tc))

def execute_evolution(symbol: str, df):
    # Asumimos que DF ya viene y extraemos 'close'
    closes_arr = np.ascontiguousarray(df['close'].values, dtype=np.float64)
    total_bars = len(closes_arr)
    
    logger.info(f"🚀 Iniciando Quantum Surrogate para {symbol} | Velas: {total_bars}")
    
    # ── FASE 0: WARMUP DEL SURROGATE (F2) ──
    logger.info("Fase 0: Recolectando 500 simulaciones F2 para entrenar Surrogate...")
    warmup_params = generate_random_params(500)
    
    y_pnl, y_dd, y_wr = [], [], []
    for p in warmup_params:
        res = run_fidelity(closes_arr, p[0], p[1], int(p[2]), int(p[3]), p[4], 'F2')
        y_pnl.append(res['pnl'])
        y_dd.append(res['max_dd'])
        y_wr.append(res['win_rate'])
        
    surrogate = ParetoSurrogateEnsemble()
    surrogate.train(warmup_params, np.array(y_pnl), np.array(y_dd), np.array(y_wr))
    
    # ── FASE 1: SURROGATE EXPLORATION (10M VIRTUALES) ──
    VIRTUAL_SAMPLES = 10_000_000
    logger.info(f"Fase 1: Generando {VIRTUAL_SAMPLES:,} configs virtuales. Pasando al Surrogate...")
    virtual_params = generate_random_params(VIRTUAL_SAMPLES)
    
    # Filtrar las top 10k en microsegundos
    top_10k_params, fitness_10k = surrogate.filter_promising_configs(virtual_params, top_k=10000)
    logger.info(f"✅ Surrogate completó 10M evaluaciones. Top 10,000 seleccionadas.")
    
    # ── FASE 2: FILTRADO F1 (Hyperband Early Stop) ──
    logger.info("Fase 2: Ejecutando F1 (100 velas) sobre las Top 10,000...")
    f1_results = []
    t0 = time.time()
    for i, p in enumerate(top_10k_params):
        res = run_fidelity(closes_arr, p[0], p[1], int(p[2]), int(p[3]), p[4], 'F1')
        f1_results.append(res['pnl'])
    logger.info(f"✅ F1 completado en {time.time()-t0:.2f}s.")
    
    # Seleccionar top 1,000
    top_1k_indices = np.argsort(f1_results)[::-1][:1000]
    top_1k_params = top_10k_params[top_1k_indices]
    
    # ── FASE 3: OPTIMIZACIÓN CMA-ES F3 / F4 ──
    logger.info("Fase 3: Inyectando sobrevivientes a Optuna (CMA-ES/TPE) para Validación Final (F4)")
    db_path = f'sqlite:///data/quantum_surrogate.db'
    study_name = f'QS_{symbol.replace("/", "")}'
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.CmaEsSampler(),
        storage=optuna.storages.RDBStorage(url=db_path, engine_kwargs={"connect_args": {"timeout": 60}}),
        load_if_exists=True
    )
    
    # Inyectar las mejores a Optuna
    for p in top_1k_params[:50]:
        study.enqueue_trial({
            'tp_pct': float(p[0]), 'sl_pct': float(p[1]), 'rsi_window': int(p[2]),
            'macd_fast': int(p[3]), 'trend_conf': float(p[4])
        })
        
    def objective(trial):
        tp_pct = trial.suggest_float('tp_pct', 0.005, 0.05)
        sl_pct = trial.suggest_float('sl_pct', 0.005, 0.03)
        window = trial.suggest_int('rsi_window', 10, 50)
        fast_window = trial.suggest_int('macd_fast', 3, 20)
        trend_conf = trial.suggest_float('trend_conf', 0.0001, 0.005)
        
        # Ejecutar F4
        res = run_fidelity(closes_arr, tp_pct, sl_pct, window, fast_window, trend_conf, 'F4')
        
        # Penalizar DD y max trades bajos
        if res['trades'] < 5: return -1000.0
        if res['max_dd'] > 0.05: return -500.0
        return res['pnl']

    t_opt = time.time()
    # Concurrency safe for sqlite
    study.optimize(objective, n_trials=200, n_jobs=1)
    logger.info(f"🏆 F4 Final completado en {time.time()-t_opt:.2f}s!")
    logger.info(f"👑 Best Genotype F4: {study.best_trial.params} | PnL: {study.best_value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()
    
    TARGET_COINS = ["BTC/USDT", "ETH/USDT"]
    logger.info("📡 Descargando datos masivos para memoria mapeada...")
    all_data = fetch_multi_symbol_data(TARGET_COINS, args.days, max_workers=2)
    
    for symbol in TARGET_COINS:
        if symbol in all_data and not all_data[symbol].empty:
            execute_evolution(symbol, all_data[symbol])

if __name__ == "__main__":
    main()
