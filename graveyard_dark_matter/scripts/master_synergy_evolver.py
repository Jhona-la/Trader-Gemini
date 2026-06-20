import sys
import os
import time
import json
import subprocess
from datetime import datetime
from copy import deepcopy

import optuna
import pandas as pd
import numpy as np
import gc

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import importlib.util
from utils.logger import logger

# Load configurations
from config import Config
adaptive_config_path = os.path.join(_project_root, "config", "adaptive_config.py")
spec = importlib.util.spec_from_file_location("adaptive_config", adaptive_config_path)
adaptive_config_module = importlib.util.module_from_spec(spec)
sys.modules["adaptive_config"] = adaptive_config_module
spec.loader.exec_module(adaptive_config_module)
adaptive_config = adaptive_config_module.adaptive_config

from core.omni_fitness import calculate_omni_fitness
from core.backtest_infra import fetch_multi_symbol_data
from scripts.run_god_mode_backtest import run_global_backtest

def update_config_dict(target_obj, overrides_dict):
    for key, value in overrides_dict.items():
        if isinstance(value, dict) and key in target_obj and isinstance(target_obj[key], dict):
            update_config_dict(target_obj[key], value)
        else:
            target_obj[key] = value

def generate_overrides(trial):
    """
    ESPACIO DE BÚSQUEDA CUÁNTICO (Unificación Total)
    Asumimos que TODO está mal y Optuna debe descubrir la Sinergia Perfecta.
    """
    
    # 1. Asignaciones de Capital
    micro_cap = trial.suggest_float('micro_alloc', 0.1, 0.5)
    scalp_cap = trial.suggest_float('scalp_alloc', 0.1, 0.5)
    swing_cap = trial.suggest_float('swing_alloc', 0.1, 0.5)
    
    # 2. Pesos del Consenso de Inteligencia Artificial
    w_ml = trial.suggest_float('w_ml', 0.5, 2.0)
    w_tech = trial.suggest_float('w_technical', 0.5, 2.0)
    w_phalanx = trial.suggest_float('w_phalanx', 0.0, 1.0)
    w_statarb = trial.suggest_float('w_statarb', 0.0, 1.0)
    master_thresh = trial.suggest_float('master_threshold', 1.0, 3.0)
    
    # 3. Riesgo y Exposición
    cvar_conf = trial.suggest_float('cvar_confidence', 0.90, 0.99)
    max_sector_micro = trial.suggest_float('max_sector_exposure_micro', 0.05, 0.20)
    max_sector_scalp = trial.suggest_float('max_sector_exposure_scalp', 0.10, 0.30)
    max_sector_swing = trial.suggest_float('max_sector_exposure_swing', 0.15, 0.40)
    daily_dd = trial.suggest_float('daily_drawdown_limit', 0.10, 0.35)

    # 4. Parámetros Técnicos y Patrones
    rsi_buy = trial.suggest_int('rsi_oversold', 20, 40)
    rsi_sell = trial.suggest_int('rsi_overbought', 60, 80)
    macd_fast = trial.suggest_int('macd_fast', 8, 14)
    macd_slow = trial.suggest_int('macd_slow', 21, 34)
    ema_trend = trial.suggest_int('ema_trend_window', 100, 250)
    
    # 5. Parámetros Dinámicos y Arquitectura Lógica (Booleanos)
    return {
        'BLUEPRINT_RISK': {
            'cvar_confidence': cvar_conf,
            'max_sector_exposure_micro': max_sector_micro,
            'max_sector_exposure_scalp': max_sector_scalp,
            'max_sector_exposure_swing': max_sector_swing,
            'daily_drawdown_limit': daily_dd
        },
        'BLUEPRINT_TECHNICAL': {
            'rsi_oversold': rsi_buy,
            'rsi_overbought': rsi_sell,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'ema_trend_window': ema_trend
        },
        'BLUEPRINT_PATTERN': {
            'wick_filter_strictness': trial.suggest_float('wick_filter_strictness', 1.0, 3.0),
            'consolidation_candles_min': trial.suggest_int('consolidation_candles_min', 8, 24)
        },
        'BLUEPRINT_SNIPER': {
            'volume_spike_multiplier': trial.suggest_float('volume_spike_multiplier', 1.2, 2.5),
            'absorption_threshold_pct': trial.suggest_float('absorption_threshold_pct', 0.2, 1.0)
        },
        'BLUEPRINT_OMNISCORE': {
            'w_ml': w_ml,
            'w_technical': w_tech,
            'w_phalanx': w_phalanx,
            'w_statarb': w_statarb,
            'master_threshold': master_thresh
        },
        'MATRIX_OVERRIDES': {
            'MICRO': {
                'global_horizon': {'capital_allocation_base_pct': micro_cap},
                'por_activo': {
                    'BTC': {
                        'tp_pct_default': trial.suggest_float('micro_btc_tp', 0.05, 0.5),
                        'sl_pct_default': trial.suggest_float('micro_btc_sl', 0.05, 0.2)
                    }
                }
            },
            'SCALP': {
                'global_horizon': {'capital_allocation_base_pct': scalp_cap},
                'por_activo': {
                    'BTC': {
                        'tp_pct_default': trial.suggest_float('scalp_btc_tp', 0.3, 1.5),
                        'sl_pct_default': trial.suggest_float('scalp_btc_sl', 0.1, 0.5)
                    }
                }
            },
            'SWING': {
                'global_horizon': {'capital_allocation_base_pct': swing_cap},
                'por_activo': {
                    'BTC': {
                        'tp_pct_default': trial.suggest_float('swing_btc_tp', 1.5, 4.0),
                        'sl_pct_default': trial.suggest_float('swing_btc_sl', 0.5, 2.0)
                    }
                }
            }
        },
        'LOGICAL_DNA': {
            'risk_dynamic_stops': trial.suggest_categorical('dna_risk_dynamic_stops', [True, False]),
            'sniper_volume_confirmation': trial.suggest_categorical('dna_sniper_volume', [True, False]),
            'pattern_strict_wick_filter': trial.suggest_categorical('dna_pattern_strict', [True, False]),
            'tech_use_garch': trial.suggest_categorical('dna_tech_garch', [True, False])
        }
    }

def objective(trial, all_data, symbols, days, original_matrix):
    overrides = generate_overrides(trial)
    
    # 1. Normalizar Asignación de Capital
    m_cap = overrides['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct']
    sc_cap = overrides['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct']
    sw_cap = overrides['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct']
    tot = m_cap + sc_cap + sw_cap
    if tot > 0:
        overrides['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct'] = m_cap / tot
        overrides['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct'] = sc_cap / tot
        overrides['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct'] = sw_cap / tot

    # 2. Inyectar variables lógicas a variables de entorno (simulación de DNA)
    dna = overrides.pop('LOGICAL_DNA')
    os.environ['DNA_RISK_DYNAMIC_STOPS'] = str(dna['risk_dynamic_stops'])
    os.environ['DNA_SNIPER_VOLUME'] = str(dna['sniper_volume_confirmation'])
    os.environ['DNA_PATTERN_STRICT'] = str(dna['pattern_strict_wick_filter'])
    os.environ['DNA_TECH_GARCH'] = str(dna['tech_use_garch'])
    
    # Inyectar números a Config Global en memoria para el backtest
    if not hasattr(Config, 'OmniScore'):
        Config.OmniScore = type('OmniScore', (), {})
    Config.OmniScore.w_ml = overrides['BLUEPRINT_OMNISCORE']['w_ml']
    Config.OmniScore.w_technical = overrides['BLUEPRINT_OMNISCORE']['w_technical']
    Config.OmniScore.w_phalanx = overrides['BLUEPRINT_OMNISCORE']['w_phalanx']
    Config.OmniScore.w_statarb = overrides['BLUEPRINT_OMNISCORE']['w_statarb']
    Config.OmniScore.master_threshold = overrides['BLUEPRINT_OMNISCORE']['master_threshold']

    # Restaurar a default primero
    adaptive_config.matrix = deepcopy(original_matrix)
    
    # Inyectar matriz
    update_config_dict(adaptive_config.matrix, overrides['MATRIX_OVERRIDES'])
    
    try:
        results = run_global_backtest(
            all_data=all_data,
            symbols=symbols,
            days=days,
            initial_capital=13.0,
            verbose=False,
            seed=42,
            scenario="MASTER_SYNERGY",
            mode="FULL"
        )
    except Exception as e:
        logger.error(f"Fallo en simulación Trial {trial.number}: {e}")
        return -9999.0
        
    metrics = results['metrics']
    trades = results['trades']
    
    if not trades or len(trades) == 0:
        return -9999.0
        
    pnls = np.array([t['pnl_pct'] / 100.0 for t in trades])
    win_rate = metrics['win_rate']
    max_dd = metrics['max_drawdown']
    total_trades = metrics['total_trades']
    
    # Colección de basura agresiva
    gc.collect()
    
    score = calculate_omni_fitness(
        pnls=pnls,
        win_rate=win_rate,
        max_dd=max_dd,
        trades=total_trades,
        starting_capital=13.0
    )
    
    logger.info(f"Trial {trial.number} | Score: {score:.2f} | WR: {win_rate:.1f}% | Trades: {total_trades}")
    return score

def run_compiler(json_path):
    compiler_path = os.path.join(_project_root, "tools", "architecture_compiler.py")
    if os.path.exists(compiler_path):
        logger.info(f"⚙️ [Compiler] Lanzando compilación estática basada en {json_path}")
        subprocess.run([sys.executable, compiler_path, "--json", json_path], check=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MASTER SYNERGY EVOLVER")
    parser.add_argument("--days", type=int, default=5, help="Días de backtest por trial")
    parser.add_argument("--trials", type=int, default=50, help="Iteraciones")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,SOLUSDT", help="Símbolos a operar")
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    days = args.days
    
    logger.info("=" * 60)
    logger.info("🌌 MASTER SYNERGY EVOLVER (UNIFICACIÓN TOTAL) INICIANDO...")
    logger.info("=" * 60)
    
    all_data = fetch_multi_symbol_data(symbols, days)
    original_matrix = deepcopy(adaptive_config.matrix)
    
    study_name = f"synergy_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    study.optimize(lambda trial: objective(trial, all_data, symbols, days, original_matrix), n_trials=args.trials)
    
    best_trial = study.best_trial
    logger.info(f"💎 Sinergia Máxima Encontrada (Score: {best_trial.value})")
    
    # Reconstruir la configuración ganadora usando los parámetros de Optuna
    overrides = generate_overrides(optuna.trial.FixedTrial(best_trial.params))
    
    # Restaurar y aplicar LOGICAL_DNA para el blueprint general
    dna = overrides['LOGICAL_DNA']
    
    # Guardar blueprint
    output_path = os.path.join(_project_root, "data", f"blueprint_master_{study_name}.json")
    
    final_output = overrides
    final_output['LOGICAL_DNA'] = dna 
    
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    logger.info(f"💾 Blueprint Master guardado en {output_path}")
    
    # Guardar también como omni_evolver para retrocompatibilidad con compiler
    omni_path = os.path.join(_project_root, "data", f"omni_evolver_best_{study_name}.json")
    with open(omni_path, "w") as f:
        # El compiler de arquitectura espera un diccionario plano con claves 'dna_*'
        json.dump(best_trial.params, f, indent=4)
        
    # Auto-Compilar Arquitectura
    run_compiler(omni_path)
    
    logger.info("✅ EVOLUCIÓN COMPLETA. Blueprint cargado y Arquitectura compilada.")

if __name__ == "__main__":
    main()
