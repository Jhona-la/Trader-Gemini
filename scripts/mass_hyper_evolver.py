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
    omni_master_threshold = trial.suggest_float('omni_master_threshold', 0.5, 2.5, step=0.1)
    omni_w_tech = trial.suggest_float('omni_w_tech', 0.5, 2.0, step=0.1)
    omni_w_phalanx = trial.suggest_float('omni_w_phalanx', 0.1, 1.0, step=0.1)
    omni_w_statarb = trial.suggest_float('omni_w_statarb', 0.1, 1.0, step=0.1)
    rsi_buy = trial.suggest_int('rsi_buy', 25, 45, step=5)
    rsi_sell = trial.suggest_int('rsi_sell', 55, 75, step=5)
    cooldown_seconds = trial.suggest_int('cooldown_seconds', 10, 300, step=10)
    
    # 2. Riesgo y Cierres
    sl_pct = trial.suggest_float('sl_pct', 0.0005, 0.0060, step=0.0005)
    tp_sl_ratio = trial.suggest_float('tp_sl_ratio', 1.0, 5.0, step=0.5)
    tp_pct = sl_pct * tp_sl_ratio
    
    # 3. Zombie-Chaser (Novedad)
    zombie_chaser_atr_mult = trial.suggest_float('zombie_chaser_atr_mult', 0.2, 2.0, step=0.1)
    
    # 4. Compounding & Risk Multipliers (Phase 14)
    ml_kelly_fraction = trial.suggest_float('ml_kelly_fraction', 0.5, 1.5, step=0.1)
    compounding_growth_factor = trial.suggest_float('compounding_growth_factor', 0.1, 0.8, step=0.05)
    
    # 5. System-Wide Risk Management (Phase 16 - User Request)
    # Permite a la IA descubrir el límite de concurrencia y tolerancia a pérdidas del sistema completo
    max_concurrent_trades = trial.suggest_int('max_concurrent_trades', 1, 5)
    max_drawdown_limit = trial.suggest_float('max_drawdown_limit', 0.02, 0.10, step=0.01)
    global_stop_loss_pct = trial.suggest_float('global_stop_loss_pct', 0.05, 0.20, step=0.01)
    
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
            'rsi_buy': rsi_buy,
            'rsi_sell': rsi_sell,
            'cooldown_seconds': cooldown_seconds
        })
        
        if not hasattr(Config, 'OmniScore'):
            Config.OmniScore = type('OmniScore', (), {})
            
        _orig_omni_master = getattr(Config.OmniScore, 'master_threshold', 1.5)
        _orig_omni_tech = getattr(Config.OmniScore, 'w_technical', 1.0)
        _orig_omni_phalanx = getattr(Config.OmniScore, 'w_phalanx', 0.5)
        _orig_omni_statarb = getattr(Config.OmniScore, 'w_statarb', 0.5)
        
        Config.OmniScore.master_threshold = omni_master_threshold
        Config.OmniScore.w_technical = omni_w_tech
        Config.OmniScore.w_phalanx = omni_w_phalanx
        Config.OmniScore.w_statarb = omni_w_statarb
        
        if not hasattr(Config.Strategies, 'Mutations'):
            Config.Strategies.Mutations = {}
        Config.Strategies.Mutations['zombie_chaser_atr_mult'] = zombie_chaser_atr_mult
        
        # [PHASE 8 & 14] Activar Quantum Compounding para que el Target Exponencial sea medible
        _orig_kelly = getattr(Config.Risk, 'ML_KELLY_FRACTION', 1.0)
        _orig_comp_growth = getattr(Config.Risk, 'COMPOUNDING_GROWTH_FACTOR', 0.30)
        _orig_max_concurrent = getattr(Config.Risk, 'MAX_CONCURRENT_TRADES_TOTAL', 4)
        _orig_max_dd = getattr(Config.Risk, 'MAX_DRAWDOWN_LIMIT', 0.05)
        _orig_global_sl = getattr(Config.Risk, 'GLOBAL_STOP_LOSS_PCT', 0.10)
        
        Config.Risk.COMPOUNDING_ENABLED = True
        Config.Risk.ML_KELLY_FRACTION = ml_kelly_fraction
        Config.Risk.COMPOUNDING_GROWTH_FACTOR = compounding_growth_factor
        Config.Risk.MAX_CONCURRENT_TRADES_TOTAL = max_concurrent_trades
        Config.Risk.MAX_DRAWDOWN_LIMIT = max_drawdown_limit
        Config.Risk.GLOBAL_STOP_LOSS_PCT = global_stop_loss_pct
        
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
                isolated_strategy="omni"
            )
        logging.disable(logging.NOTSET)
        logging.disable(logging.NOTSET)
    except Exception as e:
        logging.disable(logging.NOTSET)
        logger.error(f"❌ Error trial: {e}")
        import traceback
        traceback.print_exc()
        return -1000.0
    finally:
        logging.disable(logging.NOTSET)
        config_dict.update(_orig_config)
        
        Config.OmniScore.master_threshold = _orig_omni_master
        Config.OmniScore.w_technical = _orig_omni_tech
        Config.OmniScore.w_phalanx = _orig_omni_phalanx
        Config.OmniScore.w_statarb = _orig_omni_statarb
        
        Config.Risk.ML_KELLY_FRACTION = _orig_kelly
        Config.Risk.COMPOUNDING_GROWTH_FACTOR = _orig_comp_growth
        Config.Risk.MAX_CONCURRENT_TRADES_TOTAL = _orig_max_concurrent
        Config.Risk.MAX_DRAWDOWN_LIMIT = _orig_max_dd
        Config.Risk.GLOBAL_STOP_LOSS_PCT = _orig_global_sl

    # ── EVALUAR FITNESS COMPUESTO (QUANTUM EXPONENTIAL TARGET) ──
    metrics = result.get('metrics', {})
    trades = metrics.get('total_trades', 0)
    pnl_usd = metrics.get('final_capital', 13.0) - 13.0
    max_dd = metrics.get('max_drawdown_pct', 0) / 100.0
    win_rate = metrics.get('win_rate', 0)
    
    # Penalizaciones (Survival Filters)
    if trades < 3:
        score = -500.0 + trades  # Demasiado conservador
    elif max_dd > 0.08:
        score = -1000.0 * max_dd  # Hard Cap: Max Drawdown 8%
    
    # Target de crecimiento: 100% de ROI cada 3 días
    target_capital = 13.0 * (2 ** (days / 3.0))
    target_pnl = target_capital - 13.0

    # User Mandate: WR no necesita ser 100%, pero el PnL debe crecer de forma compuesta (100% cada 3 días)
    if win_rate < 50.0:
        score = -200.0 + win_rate # Castigo para estrategias que sangran capital
    else:
        # El objetivo principal es el hiper-crecimiento exponencial del PnL Neto
        score = pnl_usd
        
        # Super-bonificación si se logra el objetivo de 100% ROI cada 3 días
        if pnl_usd >= target_pnl:
            score += 5000.0 + (pnl_usd - target_pnl) * 2.0
            
        # Pequeño bonus para estabilizar la selección entre PnLs similares
        if win_rate >= 80:
            score += 10.0
        if max_dd < 0.02:
            score += 5.0
    
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
    # [PHASE 15 ISOLATION]
    # Set unique ENV_ID for this specific optimization thread so it doesn't lock DBs
    env_id = f"evo_{symbol.replace('/', '')}_{horizon}"
    os.environ["TG_ENV_ID"] = env_id
    Config.ENV_ID = env_id
    Config.DATA_DIR = f"dashboard/data/futures_{env_id}"
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    logger.info(f"🧬 Iniciando Evolución Masiva para {symbol} | Horizonte: {horizon} | {days} Días | {n_trials} Trials")
    
    study_name = f'evo_{symbol.replace("/", "")}_{horizon}_{days}D_V14'
    # Use global Optuna DB so all workers share studies
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
        'ml_kelly_fraction': best.params.get('ml_kelly_fraction'),
        'compounding_growth_factor': best.params.get('compounding_growth_factor'),
        'max_concurrent_trades': best.params.get('max_concurrent_trades'),
        'max_drawdown_limit': best.params.get('max_drawdown_limit'),
        'global_stop_loss_pct': best.params.get('global_stop_loss_pct'),
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
        'ZombieMulti': best.params.get('zombie_chaser_atr_mult'),
        'Kelly_Frac': best.params.get('ml_kelly_fraction'),
        'Comp_Factor': best.params.get('compounding_growth_factor'),
        'MaxConcurrent': best.params.get('max_concurrent_trades'),
        'MaxDDLimit': best.params.get('max_drawdown_limit'),
        'GlobalSLPct': best.params.get('global_stop_loss_pct')
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
    
    # Top 3 Volatile/Liquid Coins para Scalping (Phase 14 Fast Discovery)
    TARGET_COINS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT"
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
