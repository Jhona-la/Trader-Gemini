#!/usr/bin/env python3
"""
===============================================================================
 MASS HYPER EVOLVER (OMNISCIENCIA 30D)
===============================================================================
QUÉ: Una suite de Optimización Bayesiana Masiva (Optuna) que aísla cada moneda,
     simula 30 días de historia real, muta decenas de parámetros y busca el
     ADN perfecto (genotipo) que genera rentabilidades masivas.
POR QUÉ: "Una talla no sirve para todos". El ATR de BTC y PEPE requieren configuraciones
     cuánticamente distintas de Take Profit, Stop Loss, y Hurdle del ML.
PARA QUÉ: Lograr el 100% de WinRate en Scalping aislando el comportamiento específico.
"""

import os
import sys
import json
import optuna
import logging
import gc
import io
import contextlib
import time
import psutil
from datetime import datetime

# Hardware Optimization
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# Root path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from scripts.run_god_mode_backtest import run_global_backtest
from core.backtest_infra import fetch_multi_symbol_data

# Suppress Optuna logging to prevent spam
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("MassEvolver")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def objective(trial, all_data, symbol, days, horizon):
    start_time = time.time()
    
    # ── ESPACIO DE MUTACIÓN ──
    # 1. Técnico & ML
    strength_threshold = trial.suggest_float('strength_threshold', 0.25, 0.75, step=0.05)
    ml_confidence = trial.suggest_float('ml_confidence', 0.40, 0.75, step=0.05)
    adx_threshold = trial.suggest_int('adx_threshold', 15, 40)
    cooldown_seconds = trial.suggest_int('cooldown_seconds', 10, 300, step=10)
    
    # 2. Riesgo y Cierres
    sl_pct = trial.suggest_float('sl_pct', 0.0005, 0.0060, step=0.0005)
    tp_sl_ratio = trial.suggest_float('tp_sl_ratio', 1.0, 5.0, step=0.5)
    tp_pct = sl_pct * tp_sl_ratio
    
    # 3. Zombie-Chaser (Novedad)
    zombie_chaser_atr_mult = trial.suggest_float('zombie_chaser_atr_mult', 0.2, 2.0, step=0.1)
    
    # ── INYECTAR EN CONFIG GLOBAL ──
    if horizon == "SCALPING":
        config_dict = Config.Horizons.Scalping
    else:
        config_dict = Config.Horizons.Swing
        
    _orig_config = config_dict.copy()
    _orig_ml_conf = Config.Strategies.ML_MIN_CONFIDENCE
    
    try:
        config_dict.update({
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'strength_threshold': strength_threshold,
            'adx_threshold': adx_threshold,
            'cooldown_seconds': cooldown_seconds
        })
        
        Config.Strategies.ML_MIN_CONFIDENCE = ml_confidence
        
        if not hasattr(Config.Strategies, 'Mutations'):
            Config.Strategies.Mutations = {}
        Config.Strategies.Mutations['zombie_chaser_atr_mult'] = zombie_chaser_atr_mult
        
        # MUDAR AL MOTOR: Evitar que haga spam de prints y crashee la terminal
        logging.disable(logging.CRITICAL)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Ejecutar God Mode Sincronizado para 1 solo símbolo (Micro-Cuenta $13 aislada)
            result = run_global_backtest(
                all_data=all_data,
                symbols=[symbol],
                days=days,
                initial_capital=13.0,
                verbose=False,
                isolated_strategy="technical"
            )
        logging.disable(logging.NOTSET)
    except Exception as e:
        logger.error(f"❌ Error trial: {e}")
        return -1000.0
    finally:
        # Restaurar configuración original IN-PLACE
        config_dict.update(_orig_config)
        Config.Strategies.ML_MIN_CONFIDENCE = _orig_ml_conf

    # ── EVALUAR FITNESS COMPUESTO ──
    metrics = result.get('metrics', {})
    trades = metrics.get('total_trades', 0)
    pnl_usd = metrics.get('final_capital', 13.0) - 13.0
    max_dd = metrics.get('max_drawdown_pct', 0) / 100.0
    win_rate = metrics.get('win_rate', 0)
    
    # Penalizaciones (Survival Filters)
    if trades < 5:
        score = -500.0 + trades  # Demasiado conservador, no aprovecha oportunidades
    elif max_dd > 0.05:
        score = -1000.0 * max_dd  # Hard Cap: Max Drawdown 5% (Ideal < 1.5%)
    elif win_rate < 50.0:
        score = -200.0 + win_rate # Castigar debajo de coin-flip
    else:
        dd_penalty = max(0.001, max_dd)
        calmar = pnl_usd / (dd_penalty * 13.0)
        
        wr_bonus = 0
        if win_rate >= 80:
            wr_bonus = (win_rate - 80) * 10
        if win_rate >= 95:
            wr_bonus += 500  # Santo Grial
            
        score = (pnl_usd * 50) + calmar + wr_bonus + (trades * 0.1)
    
    # Guardar atributos de monitoreo
    trial.set_user_attr('trades', trades)
    trial.set_user_attr('win_rate', win_rate)
    trial.set_user_attr('pnl_usd', pnl_usd)
    trial.set_user_attr('max_dd', max_dd * 100)
    
    end_time = time.time()
    trial.set_user_attr('duration_ms', (end_time - start_time) * 1000)
    
    # Garbage collection intermitente para la memoria del SQLite y objetos Python
    if trial.number > 0 and trial.number % 50 == 0:
        gc.collect()
    
    return score

def optimize_coin(symbol, all_data, days, n_trials, horizon):
    logger.info(f"🧬 Iniciando Evolución Masiva para {symbol} | Horizonte: {horizon} | {days} Días | {n_trials} Trials")
    
    study_name = f'evo_{symbol.replace("/", "")}_{horizon}_{days}D'
    db_path = f'sqlite:///data/mass_evolver.db'
    
    # Algoritmo similar a Random Forest: Tree-structured Parzen Estimator (Multivariado)
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10)
    
    storage = optuna.storages.RDBStorage(
        url=db_path,
        engine_kwargs={"connect_args": {"timeout": 60}}
    )
    
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
        ram_mb = get_ram_usage()
        # Usamos \r para sobreescribir la misma línea en la terminal sin hacer scroll infinito
        msg = f"\r  🧬 [{symbol}][{horizon}] Trial {trial.number:5d} | Score: {trial.value or -9999:8.1f} | TR: {attrs.get('trades',0):3d} | WR: {attrs.get('win_rate',0):5.1f}% | PnL: ${attrs.get('pnl_usd',0):+.2f} | ⚡ {duration:.1f} ms/trial | RAM: {ram_mb:.1f} MB      "
        sys.stdout.write(msg)
        sys.stdout.flush()
        
    study.optimize(
        lambda t: objective(t, all_data, symbol, days, horizon),
        n_trials=n_trials,
        n_jobs=1,  # Sequencial para evitar Singletons cruzados en memoria de Python
        callbacks=[callback],
        show_progress_bar=False
    )
    
    sys.stdout.write("\n")  # Nueva línea al terminar
    best = study.best_trial
    logger.info(f"🏆 [{symbol}][{horizon}] MEJOR GENOTIPO ENCONTRADO (Trial #{best.number}):")
    logger.info(f"   Score: {best.value:.2f} | PnL: ${best.user_attrs['pnl_usd']:.2f} | WR: {best.user_attrs['win_rate']:.1f}%")
    
    # Exportar Genoma
    genotype = {
        'symbol': symbol,
        'horizon': horizon,
        'sl_pct': best.params.get('sl_pct'),
        'tp_sl_ratio': best.params.get('tp_sl_ratio'),
        'tp_pct': best.params.get('sl_pct') * best.params.get('tp_sl_ratio'),
        'strength_threshold': best.params.get('strength_threshold'),
        'ml_confidence': best.params.get('ml_confidence'),
        'adx_threshold': best.params.get('adx_threshold'),
        'cooldown_seconds': best.params.get('cooldown_seconds'),
        'zombie_chaser_atr_mult': best.params.get('zombie_chaser_atr_mult'),
        'performance': {
            'win_rate': best.user_attrs['win_rate'],
            'pnl_usd': best.user_attrs['pnl_usd'],
            'trades': best.user_attrs['trades'],
            'max_dd_pct': best.user_attrs['max_dd']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('config/genotypes', exist_ok=True)
    out_path = f'config/genotypes/{symbol.replace("/", "_")}_{horizon}_perfect_genome.yaml'
    
    import yaml
    with open(out_path, 'w') as f:
        yaml.dump(genotype, f, default_flow_style=False, sort_keys=False)
        
    # CLASIFICACIÓN GLOBAL EN PARQUET
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    parquet_path = 'config/genotypes/global_evolution_report.parquet'
    
    new_row = pd.DataFrame([{
        'Symbol': symbol, 
        'Horizon': horizon,
        'Score': round(best.value, 2), 
        'WinRate_Pct': round(best.user_attrs['win_rate'], 1), 
        'PnL_USD': round(best.user_attrs['pnl_usd'], 2),
        'Max_DD_Pct': round(best.user_attrs['max_dd']*100, 2), 
        'Trades': best.user_attrs['trades'], 
        'SL_Pct': best.params.get('sl_pct'), 
        'TP_Pct': best.params.get('sl_pct') * best.params.get('tp_sl_ratio'),
        'Strength': best.params.get('strength_threshold'), 
        'ADX': best.params.get('adx_threshold'),
        'Cooldown': best.params.get('cooldown_seconds'), 
        'ZombieMulti': best.params.get('zombie_chaser_atr_mult')
    }])
    
    if os.path.exists(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        df_combined = pd.concat([df_existing, new_row], ignore_index=True)
        df_combined = df_combined.sort_values('Score', ascending=False).drop_duplicates(subset=['Symbol', 'Horizon'], keep='first')
        df_combined.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    else:
        new_row.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        
    logger.info(f"💾 Genoma guardado en {out_path} y clasificado en Parquet.\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=15, help='Días de histórico a descargar')
    parser.add_argument('--trials', type=int, default=100, help='Trials por moneda')
    args = parser.parse_args()
    
    # Top 10 Volatile/Liquid Coins para Scalping
    TARGET_COINS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", 
        "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", 
        "LINK/USDT", "PEPE/USDT"
    ]
    HORIZONS = ["SCALPING", "SWING"]
    
    logger.info("======================================================")
    logger.info("🔥 MASS HYPER EVOLVER INICIALIZADO 🔥")
    logger.info(f"   Monedas: {len(TARGET_COINS)}")
    logger.info(f"   Horizontes: SCALPING, SWING")
    logger.info(f"   Días: {args.days} (~ {args.days*1440} barras por moneda)")
    logger.info(f"   Trials/Horizonte: {args.trials}")
    logger.info("======================================================")
    
    logger.info("\n📡 Descargando Data RAM (Operación Intensiva)...")
    all_data_raw = fetch_multi_symbol_data(TARGET_COINS, args.days, max_workers=4)
    
    if not all_data_raw:
        logger.error("❌ Falla crítica en la recolección de datos.")
        sys.exit(1)
        
    logger.info("⚡ Pre-Estructurando Data a NumPy (Quantum Speed)...")
    from core.backtest_infra import BacktestDataProvider
    from queue import Queue
    temp_dp = BacktestDataProvider(Queue(), TARGET_COINS, all_data_raw)
    all_data_structured = temp_dp.struct_data
    
    # Destruir referencias antiguas para liberar RAM
    del temp_dp
    del all_data_raw
    gc.collect()
        
    logger.info("✅ Data estructurada. Comenzando iteraciones genéticas.\n")
    
    for symbol in TARGET_COINS:
        for horizon in HORIZONS:
            optimize_coin(symbol, all_data_structured, args.days, args.trials, horizon)
            gc.collect()
            
    logger.info("🎉 EVOLUCIÓN MASIVA COMPLETADA. Revisa config/genotypes/")
