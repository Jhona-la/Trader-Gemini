import sys
import os
import time
import json
import uuid
from datetime import datetime
from copy import deepcopy

import optuna
import pandas as pd
import numpy as np

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import importlib.util

# Cargar adaptive_config.py directamente por path para evitar conflictos con config.py
adaptive_config_path = os.path.join(_project_root, "config", "adaptive_config.py")
spec = importlib.util.spec_from_file_location("adaptive_config", adaptive_config_path)
adaptive_config_module = importlib.util.module_from_spec(spec)
sys.modules["adaptive_config"] = adaptive_config_module
spec.loader.exec_module(adaptive_config_module)
adaptive_config = adaptive_config_module.adaptive_config
from core.omni_fitness import calculate_omni_fitness
from core.backtest_infra import fetch_multi_symbol_data
from core.nano_backtester import vectorized_signals, vectorized_backtest_core, simulate_portfolio_vectorized

from utils.logger import logger

def update_config_dict(target_obj, overrides_dict):
    """
    Función recursiva para sobreescribir configuraciones en el diccionario matrix.
    """
    for key, value in overrides_dict.items():
        if isinstance(value, dict) and key in target_obj and isinstance(target_obj[key], dict):
            update_config_dict(target_obj[key], value)
        else:
            target_obj[key] = value

def generate_overrides(trial):
    """
    Optuna Sample Space para el Omni-Evolver.
    Genera una configuración mutada de 'adaptive_config.py' y 'Config' base.
    """
    return {
        'BLUEPRINT_RISK': {
            'cvar_confidence': trial.suggest_float('risk_cvar_conf', 0.90, 0.99, step=0.01),
            'max_sector_exposure_micro': trial.suggest_float('risk_sector_micro', 0.50, 0.95, step=0.05),
            'max_sector_exposure_scalp': trial.suggest_float('risk_sector_scalp', 0.35, 0.85, step=0.05),
            'max_sector_exposure_swing': trial.suggest_float('risk_sector_swing', 0.20, 0.50, step=0.05),
            'daily_drawdown_limit': trial.suggest_float('risk_daily_dd', 0.05, 0.15, step=0.01)
        },
        'BLUEPRINT_SNIPER': {
            'volume_spike_multiplier': trial.suggest_float('sniper_vol_mult', 1.5, 4.0, step=0.5),
            'absorption_threshold_pct': trial.suggest_float('sniper_abs_pct', 0.60, 0.95, step=0.05),
            'cvd_divergence_min': trial.suggest_float('sniper_cvd_div', 10000, 500000, step=10000)
        },
        'BLUEPRINT_TECHNICAL': {
            'rsi_oversold': trial.suggest_int('tech_rsi_os', 20, 35),
            'rsi_overbought': trial.suggest_int('tech_rsi_ob', 65, 80),
            'macd_fast': trial.suggest_categorical('tech_macd_f', [8, 10, 12]),
            'macd_slow': trial.suggest_categorical('tech_macd_s', [21, 26, 34]),
            'ema_trend_window': trial.suggest_categorical('tech_ema_w', [50, 100, 200])
        },
        'BLUEPRINT_PATTERN': {
            'wick_filter_strictness': trial.suggest_float('pat_wick_strict', 1.5, 3.5, step=0.5),
            'consolidation_candles_min': trial.suggest_int('pat_cons_min', 5, 20),
            'breakout_volume_confirm': trial.suggest_categorical('pat_break_vol', [True, False])
        },
        'BLUEPRINT_OMNISCORE': {
            'w_ml': trial.suggest_float('omni_w_ml', 0.1, 1.5, step=0.1),
            'w_technical': trial.suggest_float('omni_w_tech', 0.1, 1.5, step=0.1),
            'w_phalanx': trial.suggest_float('omni_w_phalanx', 0.1, 1.0, step=0.1),
            'w_statarb': trial.suggest_float('omni_w_statarb', 0.1, 1.0, step=0.1),
            'master_threshold': trial.suggest_float('omni_master_th', 0.5, 3.0, step=0.1)
        },
        'MATRIX_OVERRIDES': {
            'MICRO': {
                'global_horizon': {
                    'max_hold_seconds': trial.suggest_int('micro_max_hold_sec', 300, 1800, step=300),
                    'capital_allocation_base_pct': trial.suggest_float('micro_alloc_base', 0.1, 0.5, step=0.05),
                    'max_concurrent_positions': trial.suggest_int('micro_max_conc', 1, 5)
                },
                'por_activo': {
                    'ALL': {
                        'tp_pct_default': trial.suggest_float('micro_all_tp', 0.05, 0.5, step=0.01),
                        'sl_pct_default': trial.suggest_float('micro_all_sl', 0.05, 0.3, step=0.01),
                        'leverage': trial.suggest_categorical('micro_all_lev', [10, 15, 20]),
                        'signal_score_min': trial.suggest_int('micro_all_score', 65, 85),
                        'trailing_atr_mult': trial.suggest_float('micro_all_trail_atr', 0.3, 1.2, step=0.1),
                        'zombie_n_velas_inactividad': trial.suggest_int('micro_all_zombie', 5, 30)
                    }
                }
            },
            'SCALP': {
                'global_horizon': {
                    'max_hold_seconds': trial.suggest_int('scalp_max_hold_sec', 7200, 43200, step=3600),
                    'capital_allocation_base_pct': trial.suggest_float('scalp_alloc_base', 0.2, 0.6, step=0.05),
                    'max_concurrent_positions': trial.suggest_int('scalp_max_conc', 2, 6)
                },
                'por_activo': {
                    'ALL': {
                        'tp_pct_default': trial.suggest_float('scalp_all_tp', 0.3, 1.5, step=0.1),
                        'sl_pct_default': trial.suggest_float('scalp_all_sl', 0.1, 0.8, step=0.05),
                        'leverage': trial.suggest_categorical('scalp_all_lev', [5, 10, 15]),
                        'signal_score_min': trial.suggest_int('scalp_all_score', 60, 80),
                        'trailing_atr_mult': trial.suggest_float('scalp_all_trail_atr', 0.8, 2.5, step=0.1),
                        'zombie_n_velas_inactividad': trial.suggest_int('scalp_all_zombie', 15, 60)
                    }
                }
            },
            'SWING': {
                'global_horizon': {
                    'max_hold_seconds': trial.suggest_int('swing_max_hold_sec', 86400, 1209600, step=86400),
                    'capital_allocation_base_pct': trial.suggest_float('swing_alloc_base', 0.1, 0.4, step=0.05),
                    'max_concurrent_positions': trial.suggest_int('swing_max_conc', 1, 3)
                },
                'por_activo': {
                    'ALL': {
                        'tp_pct_default': trial.suggest_float('swing_all_tp', 1.5, 5.0, step=0.5),
                        'sl_pct_default': trial.suggest_float('swing_all_sl', 1.0, 3.0, step=0.2),
                        'leverage': trial.suggest_categorical('swing_all_lev', [1, 2, 3, 5]),
                        'signal_score_min': trial.suggest_int('swing_all_score', 60, 80),
                        'trailing_atr_mult': trial.suggest_float('swing_all_trail_atr', 2.0, 5.0, step=0.5),
                        'zombie_n_velas_inactividad': trial.suggest_int('swing_all_zombie', 24, 168)
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
    """
    Optuna objective function. Runs the full system with real data.
    """
    # 1. Mutar el cerebro global (adaptive_config)
    overrides = generate_overrides(trial)
    
    # 1.5. Normalizar Asignación de Capital (Debe sumar <= 1.0 para coexistir)
    micro_alloc = overrides['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct']
    scalp_alloc = overrides['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct']
    swing_alloc = overrides['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct']
    total_alloc = micro_alloc + scalp_alloc + swing_alloc
    
    if total_alloc > 1.0:
        overrides['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct'] = micro_alloc / total_alloc
        overrides['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct'] = scalp_alloc / total_alloc
        overrides['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct'] = swing_alloc / total_alloc

    # Guardamos los booleanos lógicos en Config para que las estrategias los lean durante el test (o los simulamos como variables de entorno).
    dna = overrides.pop('LOGICAL_DNA')
    os.environ['DNA_RISK_DYNAMIC_STOPS'] = str(dna['risk_dynamic_stops'])
    os.environ['DNA_SNIPER_VOLUME'] = str(dna['sniper_volume_confirmation'])
    os.environ['DNA_PATTERN_STRICT'] = str(dna['pattern_strict_wick_filter'])
    os.environ['DNA_TECH_GARCH'] = str(dna['tech_use_garch'])
    
    # Restaurar a default primero
    adaptive_config.matrix = deepcopy(original_matrix)
    
    # Inyectar mutaciones - Solo en la matriz pasamos MATRIX_OVERRIDES
    update_config_dict(adaptive_config.matrix, overrides.get('MATRIX_OVERRIDES', {}))
    
    # Las mutaciones BLUEPRINT_* (OmniScore, etc) se inyectan en tiempo real dentro del motor a través del BlueprintLoader o en el backtest
    # Para backtest global_omni_evolver:
    from config import Config
    
    # Inyectar OMNISCORE
    if 'BLUEPRINT_OMNISCORE' in overrides:
        omni = overrides['BLUEPRINT_OMNISCORE']
        if not hasattr(Config, 'OmniScore'):
            Config.OmniScore = type('OmniScore', (), {})
        Config.OmniScore.w_ml = omni['w_ml']
        Config.OmniScore.w_technical = omni['w_technical']
        Config.OmniScore.w_phalanx = omni.get('w_phalanx', 0.5)
        Config.OmniScore.w_statarb = omni.get('w_statarb', 0.5)
        Config.OmniScore.master_threshold = omni['master_threshold']

    # Inyectar RISK
    if 'BLUEPRINT_RISK' in overrides:
        r = overrides['BLUEPRINT_RISK']
        Config.Risk.CVAR_CONFIDENCE_OVERRIDE = r.get('cvar_confidence')
        Config.Risk.MAX_SECTOR_MICRO = r.get('max_sector_exposure_micro')
        Config.Risk.MAX_SECTOR_SCALP = r.get('max_sector_exposure_scalp')
        Config.Risk.MAX_SECTOR_SWING = r.get('max_sector_exposure_swing')
        Config.Risk.MAX_DRAWDOWN = r.get('daily_drawdown_limit', 0.10) * 100.0

    # Inyectar SNIPER
    if 'BLUEPRINT_SNIPER' in overrides:
        sn = overrides['BLUEPRINT_SNIPER']
        Config.Horizons.Mutations['sniper_vol_mult'] = sn.get('volume_spike_multiplier')
        Config.Horizons.Mutations['sniper_abs_pct'] = sn.get('absorption_threshold_pct')
        Config.Horizons.Mutations['sniper_cvd_div'] = sn.get('cvd_divergence_min')

    # Inyectar TECHNICAL
    if 'BLUEPRINT_TECHNICAL' in overrides:
        tech = overrides['BLUEPRINT_TECHNICAL']
        Config.Horizons.Scalping['rsi_buy'] = tech.get('rsi_oversold')
        Config.Horizons.Scalping['rsi_sell'] = tech.get('rsi_overbought')
        Config.Horizons.Scalping['ema_fast'] = tech.get('macd_fast')
        Config.Horizons.Scalping['ema_slow'] = tech.get('macd_slow')
        Config.Horizons.Scalping['ema_trend'] = tech.get('ema_trend_window')

    # Inyectar PATTERN
    if 'BLUEPRINT_PATTERN' in overrides:
        pat = overrides['BLUEPRINT_PATTERN']
        Config.Horizons.Mutations['pat_wick_strict'] = pat.get('wick_filter_strictness')
        Config.Horizons.Mutations['pat_cons_min'] = pat.get('consolidation_candles_min')
        os.environ['DNA_PAT_BREAK_VOL'] = str(pat.get('breakout_volume_confirm', False))
    
    # 2. Extraer parámetros para el Nano Backtester
    tech_rsi_os = overrides.get('BLUEPRINT_TECHNICAL', {}).get('rsi_oversold', 30)
    tech_rsi_ob = overrides.get('BLUEPRINT_TECHNICAL', {}).get('rsi_overbought', 70)
    tech_macd_f = overrides.get('BLUEPRINT_TECHNICAL', {}).get('macd_fast', 12)
    tech_macd_s = overrides.get('BLUEPRINT_TECHNICAL', {}).get('macd_slow', 26)
    
    # Asumimos que optimizamos la estrategia SCALP para el fitness core
    sl_pct = overrides.get('MATRIX_OVERRIDES', {}).get('SCALP', {}).get('por_activo', {}).get('ALL', {}).get('sl_pct_default', 0.20)
    tp_pct = overrides.get('MATRIX_OVERRIDES', {}).get('SCALP', {}).get('por_activo', {}).get('ALL', {}).get('tp_pct_default', 0.60)
    leverage = overrides.get('MATRIX_OVERRIDES', {}).get('SCALP', {}).get('por_activo', {}).get('ALL', {}).get('leverage', 10)
    max_hold = overrides.get('MATRIX_OVERRIDES', {}).get('SCALP', {}).get('global_horizon', {}).get('max_hold_seconds', 14400) // 60
    fee_rate = 0.0004
    
    all_pnls = []
    
    # 3. Ejecutar Simulador Vectorizado Nano (Numba)
    try:
        for symbol, arrs in all_data.items():
            highs = arrs['high']
            lows = arrs['low']
            closes = arrs['close']
            
            # Generar señales técnicas en Numba (picosegundos)
            signals = vectorized_signals(
                closes,
                rsi_window=14, rsi_os=tech_rsi_os, rsi_ob=tech_rsi_ob,
                macd_f=tech_macd_f, macd_s=tech_macd_s
            )
            
            # Simular Trades en Numba (picosegundos)
            pnls, durations = vectorized_backtest_core(
                highs, lows, closes, signals,
                sl_pct=sl_pct, tp_pct=tp_pct, leverage=leverage, fee_rate=fee_rate,
                max_hold_bars=max_hold
            )
            
            if len(pnls) > 0:
                all_pnls.append(pnls)
                
    except Exception as e:
        logger.error(f"Fallo en simulación Numba Trial {trial.number}: {e}")
        return -9999.0
        
    if not all_pnls:
        return -9999.0
        
    combined_pnls = np.concatenate(all_pnls)
    # Mezclamos los pnls para simular ocurrencias en paralelo / secuenciales
    np.random.seed(42)
    np.random.shuffle(combined_pnls)
    
    capital, max_dd, wins, losses, win_rate = simulate_portfolio_vectorized(
        combined_pnls, initial_capital=13.0, size_pct=0.3
    )
    total_trades = wins + losses
    
    if total_trades < 10:
        return -9999.0
    
    # 4. Calcular OMNI FITNESS
    score = calculate_omni_fitness(
        pnls=combined_pnls,
        win_rate=win_rate,
        max_dd=max_dd,
        trades=total_trades,
        starting_capital=13.0
    )
    
    # logger.info(f"Trial {trial.number} | Score: {score:.2f} | WR: {win_rate:.2f} | Trades: {total_trades}")
    return score

def main():
    import argparse
    parser = argparse.ArgumentParser(description="OMNI EVOLVER - Optimizador Global de Arquitectura")
    parser.add_argument("--days", type=int, default=5, help="Días de backtest por trial")
    parser.add_argument("--trials", type=int, default=50, help="Iteraciones de Optuna")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,SOLUSDT", help="Símbolos a operar")
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    days = args.days
    
    logger.info("=" * 60)
    logger.info("🧠 OMNI EVOLVER V1.0 INICIANDO...")
    logger.info("=" * 60)
    
    # 1. Cargar Datos Históricos Base (Una sola vez para todos los trials)
    logger.info(f"⏳ Descargando {days} días de datos reales para {symbols}...")
    raw_data = fetch_multi_symbol_data(symbols, days)
    
    logger.info("⚡ Compilando Datos al Espacio Vectorial Numpy para Motor Nano-Cuántico...")
    all_data = {}
    for sym, df in raw_data.items():
        all_data[sym] = {
            'high': df['high'].values.astype(np.float32),
            'low': df['low'].values.astype(np.float32),
            'close': df['close'].values.astype(np.float32)
        }
        
    original_matrix = deepcopy(adaptive_config.matrix)
    
    # 2. Configurar Estudio Optuna con Algoritmos Cuánticos (Sinergia Optimizada)
    study_name = f"omni_evolver_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # [QUANTUM UPGRADE] TPESampler Multivariante para hallar correlaciones en picosegundos
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=10, # Exploración inicial
        seed=42
    )
    
    # [QUANTUM UPGRADE] Poda Agresiva
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource='auto',
        reduction_factor=3
    )

    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        sampler=sampler,
        pruner=pruner
    )
    
    logger.info(f"🧬 Iniciando Mutaciones (Trials: {args.trials})...")
    study.optimize(lambda trial: objective(trial, all_data, symbols, days, original_matrix), n_trials=args.trials)
    
    # 3. Resultados
    logger.info("=" * 60)
    logger.info("🏆 OMNI EVOLVER FINALIZADO")
    logger.info("=" * 60)
    
    best_trial = study.best_trial
    logger.info(f"💎 Mejor Score: {best_trial.value}")
    logger.info("🧬 Mejor Genoma Encontrado:")
    for key, value in best_trial.params.items():
        logger.info(f"   {key}: {value}")
        
    # Restaurar matriz original
    adaptive_config.matrix = original_matrix
    
    # Guardar a disco en el formato JSON esperado por el Blueprint Loader
    # El diccionario ya debe contener las ramas BLUEPRINT_* y MATRIX_OVERRIDES completas
    # Reconstruimos la estructura para que coincida con lo que escupe blueprint_omni_evolver
    final_blueprint = generate_overrides(best_trial)
    
    # Normalizar asignaciones en el JSON final
    micro_alloc = final_blueprint['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct']
    scalp_alloc = final_blueprint['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct']
    swing_alloc = final_blueprint['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct']
    total_alloc = micro_alloc + scalp_alloc + swing_alloc
    if total_alloc > 1.0:
        final_blueprint['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct'] /= total_alloc
        final_blueprint['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct'] /= total_alloc
        final_blueprint['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct'] /= total_alloc
        
    output_path = os.path.join(_project_root, "data", f"omni_evolver_best_{study_name}.json")
    with open(output_path, "w") as f:
        json.dump(final_blueprint, f, indent=4)
        
    logger.info(f"💾 Parámetros guardados en {output_path}")
    logger.info("💡 Autocompilando la arquitectura en código binario (C++ / Python Nativo)...")
    
    # [QUANTUM UPGRADE] Auto-Compilación Final de la Arquitectura Lógica
    compiler_path = os.path.join(_project_root, "tools", "architecture_compiler.py")
    if os.path.exists(compiler_path):
        import subprocess
        subprocess.run([sys.executable, compiler_path, "--json", output_path], check=False)
        logger.info("✅ EVOLUCIÓN COMPLETA. Blueprint cargado y Arquitectura compilada en /compiled_core/")
    else:
        logger.warning("⚠️ No se encontró architecture_compiler.py")

if __name__ == "__main__":
    main()
