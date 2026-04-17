import sys
import os
import json
import optuna
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress Optuna logs during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Project root injection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from scripts.run_god_mode_backtest import run_global_backtest
from core.backtest_infra import fetch_multi_symbol_data

def objective(trial, all_data, symbols, horizon_days, strategy_name='Technical'):
    """
    🎯 Función Objetivo Dual: Calmar Ratio con restricción de trades mínimos
    Adaptado 100% al motor God Mode Sincronizado para validación en micro-cuenta ($13)
    """
    # ── ESPACIO DE BÚSQUEDA ──────────────────────────────────────────────────
    
    # 1. SL/TP Calibration
    sl_pct = trial.suggest_float('sl_pct', 0.0010, 0.0050, step=0.0005)
    tp_sl_ratio = trial.suggest_float('tp_sl_ratio', 1.5, 4.0, step=0.25)
    tp_pct = sl_pct * tp_sl_ratio
    
    # 2. Sensibilidad
    strength_threshold = trial.suggest_float('strength_threshold', 0.40, 0.65, step=0.05)
    ml_confidence = trial.suggest_float('ml_confidence', 0.48, 0.65, step=0.02)
    
    # 3. Trailing & Scaling
    cooldown_sec = trial.suggest_int('cooldown_seconds', 15, 60, step=15)
    
    # ── INYECTAR PARÁMETROS EN CONFIGURACIÓN GLOBAL ──────────────────────────
    _orig_scalp = Config.Strategies.SCALPING_PARAMS.copy()
    _orig_swing = Config.Strategies.SWING_PARAMS.copy()
    _orig_min_conf = Config.Strategies.ML_MIN_CONFIDENCE
    
    try:
        # Parchar la configuración específica del horizonte
        if horizon_days <= 1:
            params = Config.Strategies.SCALPING_PARAMS
        else:
            params = Config.Strategies.SWING_PARAMS
            
        params['sl_pct'] = sl_pct
        params['tp_pct'] = tp_pct
        params['strength_threshold'] = strength_threshold
        params['cooldown_seconds'] = cooldown_sec
        Config.Strategies.ML_MIN_CONFIDENCE = ml_confidence
        
        # Ejecutar God Mode Sincronizado con variables actuales
        result = run_global_backtest(
            all_data=all_data,
            symbols=symbols,
            days=horizon_days,
            initial_capital=13.0, # Fuerza Micro-Cap
            verbose=False
        )
    except Exception as e:
        print(f"❌ Error durante el trial: {e}")
        return -100.0
    finally:
        # Limpieza Incondicional de la estructura mutada
        Config.Strategies.SCALPING_PARAMS = _orig_scalp
        Config.Strategies.SWING_PARAMS = _orig_swing
        Config.Strategies.ML_MIN_CONFIDENCE = _orig_min_conf
        
    # ── EVALUAR RESULTADO A PARTIR DE GOD MODE MÉTRICAS ──────────────────────
    metrics = result.get('metrics', {})
    trades = metrics.get('total_trades', 0)
    pnl_usd = metrics.get('final_capital', 13.0) - 13.0
    max_dd = metrics.get('max_drawdown_pct', 0) / 100.0 
    win_rate = metrics.get('win_rate', 0)
    
    # Restricciones Severas
    if trades < 5:
        return -50.0 + trades  # Overfitting, demasiado poco.
    
    if max_dd > 0.05:
        return -20.0 * max_dd  # Hard Cap Drawdown < 5%
    
    # Métrica Estándar: Calmar
    if max_dd > 0.001:
        calmar = pnl_usd / (max_dd * 13.0)
    elif pnl_usd > 0:
        calmar = pnl_usd * 10
    else:
        calmar = pnl_usd
        
    wr_bonus = (win_rate - 50.0) * 0.1 if win_rate > 50 else 0
    trade_bonus = min(trades / 50.0, 1.0) * 0.5
    
    score = calmar + wr_bonus + trade_bonus
    
    # Reportar Trial para Early Pruning (Interfiriendo con HyperBand si se aplica)
    trial.report(score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return score

def run_optimization(horizon_days=1, strategy='ML_XGBoost', n_trials=10, symbols=['BTC/USDT']):
    """
    🔮 Ejecuta la optimización Bayesiana asegurando paridad con God Mode. 
    """
    print("=" * 70)
    print(f"🔮 OPTUNA ORACLE TUNER v3.0 (GOD MODE) — Horizon: {horizon_days}D")
    print(f"   Estrategia Clave: {strategy}")
    print(f"   Capital Bloqueado: $13.0 USD | Meta-Trials: {n_trials}")
    print("=" * 70)
    
    # Fetch Data Bulk
    fetch_days = horizon_days + 1  # Ligeros dias extra para warmup
    print(f"\n📡 Descargando Data Sincronizada para {len(symbols)} Símbolos ({fetch_days}d)...")
    all_data = fetch_multi_symbol_data(symbols, fetch_days, max_workers=2)
    
    if not all_data:
        print("❌ Falla crítica al descargar datos.")
        return None
        
    print(f"✅ Descarga Completada para God Mode.\n")
    
    study = optuna.create_study(
        direction='maximize',
        study_name=f'god_mode_{horizon_days}D_{strategy}',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        storage=f'sqlite:///data/optuna_studies.db',
        load_if_exists=True
    )
    
    print(f"🧬 Iniciando {n_trials} ensayos en Engine Múltiple...")
    study.optimize(
        lambda trial: objective(trial, all_data, symbols, horizon_days, strategy),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Sequential to avoid Singleton Config Collisions
    )
    
    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"🏆 MEJOR CONFIGURACIÓN (TRIAL #{best.number}):")
    print(f"   Score: {best.value:.4f}")
    print(f"   Parámetros Híbridos Recuperados:")
    for k, v in best.params.items():
        print(f"     {k}: {v}")
    print(f"{'=' * 70}")
    
    profile = {
        'sl_pct': best.params.get('sl_pct', 0.0015),
        'tp_sl_ratio': best.params.get('tp_sl_ratio', 2.0),
        'strength_threshold': best.params.get('strength_threshold', 0.55),
        'ml_confidence': best.params.get('ml_confidence', 0.52),
        'cooldown_seconds': best.params.get('cooldown_seconds', 15),
        'score': best.value,
        'timestamp': datetime.now().isoformat(),
        'n_trials': n_trials,
    }
    
    out_path = f'data/oracle_profile_god_{horizon_days}D.json'
    with open(out_path, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"\n📁 Configuración Guardada para Producción en: {out_path}")
    
    return profile

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Optuna Oracle Tuner God Mode (Micro)')
    parser.add_argument('--horizon', type=int, default=1, choices=[1, 7, 15], help='Horizonte (días)')
    parser.add_argument('--strategy', type=str, default='ML_XGBoost', help='Estrategia Base')
    parser.add_argument('--trials', type=int, default=10, help='Total Iteraciones de Optimización')
    parser.add_argument('--symbols', type=str, default='BTC/USDT', help='Pares a evaluar separados por coma')
    args = parser.parse_args()
    
    syms = [s.strip() for s in args.symbols.split(',')]
    
    run_optimization(
        horizon_days=args.horizon,
        strategy=args.strategy,
        n_trials=args.trials,
        symbols=syms
    )
