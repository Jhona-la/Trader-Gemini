#!/usr/bin/env python3
"""
===============================================================================
 MASS HYPER EVOLVER (QUANTUM VECTORIZED)
===============================================================================
QUÉ: Una suite de Optimización Bayesiana Masiva (Optuna) que aísla cada moneda,
     simula la historia usando el motor de Numba JIT (Vectorizado),
     y evalúa mutaciones a millones de velas por segundo.
POR QUÉ: El backtest tradicional tardaba horas por símbolo. El motor vectorizado
     lo reduce a milisegundos por ensayo.
"""

import os
import sys
import json
import optuna
import logging
import gc
import time
import argparse
from datetime import datetime

# Hardware Optimization
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.vectorized_backtest import run_vectorized_simulation, create_feature_matrices

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("QuantumEvolver")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def objective(trial, open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr):
    start_time = time.time()
    
    # ── ESPACIO DE MUTACIÓN ──
    # Riesgo y Cierres
    sl_pct = trial.suggest_float('sl_pct', 0.0005, 0.0060, step=0.0005)
    tp_sl_ratio = trial.suggest_float('tp_sl_ratio', 1.0, 5.0, step=0.5)
    tp_pct = sl_pct * tp_sl_ratio
    
    # Técnico
    rsi_oversold = trial.suggest_int('rsi_oversold', 20, 45, step=5)
    rsi_overbought = trial.suggest_int('rsi_overbought', 55, 80, step=5)
    
    # Compounding & Risk Multipliers
    ml_kelly_fraction = trial.suggest_float('ml_kelly_fraction', 0.5, 1.5, step=0.1)
    compounding_growth_factor = trial.suggest_float('compounding_growth_factor', 0.1, 1.5, step=0.1)
    
    # ── EJECUTAR MOTOR C-LEVEL ──
    try:
        final_equity, trades, win_rate, max_dd = run_vectorized_simulation(
            open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr,
            sl_pct, tp_pct, rsi_oversold, rsi_overbought,
            ml_kelly_fraction, compounding_growth_factor
        )
    except Exception as e:
        logger.error(f"❌ Error trial: {e}")
        return -1000.0

    # ── EVALUAR FITNESS COMPUESTO ──
    pnl_usd = final_equity - 13.0
    
    if trades < 3:
        score = -500.0 + trades
    elif max_dd > 0.08:
        score = -1000.0 * max_dd
    elif win_rate < 50.0:
        score = -200.0 + win_rate
    else:
        score = pnl_usd
        if win_rate >= 80:
            score += 10.0
        if max_dd < 0.02:
            score += 5.0
            
    trial.set_user_attr('trades', trades)
    trial.set_user_attr('win_rate', win_rate)
    trial.set_user_attr('pnl_usd', pnl_usd)
    trial.set_user_attr('max_dd', max_dd * 100)
    
    end_time = time.time()
    trial.set_user_attr('duration_ms', (end_time - start_time) * 1000)
    
    return score

def optimize_coin(symbol, df, n_trials, horizon):
    logger.info(f"🧬 Precomputando matrices (Vectorización) para {symbol}...")
    t0 = time.time()
    open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr = create_feature_matrices(df)
    logger.info(f"✅ Matrices listas en {(time.time()-t0)*1000:.1f} ms.")

    logger.info(f"🧬 Iniciando Quantum Evolver para {symbol} | Horizonte: {horizon} | {n_trials} Trials")
    study_name = f'quantum_{symbol.replace("/", "")}_{horizon}_V1'
    db_path = f'sqlite:///data/mass_evolver.db'
    
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10)
    storage = optuna.storages.RDBStorage(url=db_path, engine_kwargs={"connect_args": {"timeout": 60}})
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        storage=storage,
        load_if_exists=True
    )
    
    def callback(study, trial):
        attrs = trial.user_attrs
        duration = attrs.get('duration_ms', 0)
        msg = f"\r  🧬 [{symbol}][{horizon}] Trial {trial.number:5d} | Score: {trial.value or -9999:8.1f} | TR: {attrs.get('trades',0):3d} | WR: {attrs.get('win_rate',0):5.1f}% | PnL: ${attrs.get('pnl_usd',0):+.2f} | ⚡ {duration:.3f} ms/trial      "
        sys.stdout.write(msg)
        sys.stdout.flush()

    # Pre-compile the Numba function (burn-in run)
    run_vectorized_simulation(
        open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr,
        0.005, 0.015, 30.0, 70.0, 1.0, 1.0
    )

    t_evo_start = time.time()
    study.optimize(
        lambda t: objective(t, open_arr, high_arr, low_arr, close_arr, rsi_arr, atr_arr),
        n_trials=n_trials,
        callbacks=[callback]
    )
    t_evo_end = time.time()
    
    sys.stdout.write("\n")
    logger.info(f"🏆 Optimización de {symbol} terminada en {(t_evo_end - t_evo_start):.1f} segundos!")
    
    best = study.best_trial
    logger.info(f"👑 Best Trial [{best.number}]: Score={best.value:.2f} | PnL=${best.user_attrs.get('pnl_usd'):.2f} | WR={best.user_attrs.get('win_rate'):.1f}%")
    logger.info(f"🧬 Best Genotype: {best.params}")

def run_mass_hyper_evolver(days, trials):
    # Dynamic imports for local execution context
    from core.backtest_infra import fetch_multi_symbol_data
    import pandas as pd
    
    TARGET_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    logger.info(f"📡 Descargando Data para Quantum Vectorized Engine ({days} días)...")
    
    all_data_raw = fetch_multi_symbol_data(TARGET_COINS, days, max_workers=4)
    
    for symbol in TARGET_COINS:
        logger.info(f"📥 Procesando {symbol}...")
        df = all_data_raw.get(symbol)
        if df is None or df.empty:
            logger.error(f"❌ Failed to fetch data for {symbol}")
            continue
            
        optimize_coin(symbol, df, trials, horizon="SCALPING")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Vectorized Mass Evolver")
    parser.add_argument('--days', type=int, default=3, help='Historical days to load')
    parser.add_argument('--trials', type=int, default=1000, help='Number of optuna trials per coin')
    args = parser.parse_args()
    
    run_mass_hyper_evolver(args.days, args.trials)
