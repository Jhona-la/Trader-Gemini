import os
import sys
import gc
import math
import uuid
import optuna
import logging
import argparse
from datetime import datetime

# Root paths
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from scripts.run_god_mode_backtest import run_global_backtest
from core.backtest_infra import fetch_multi_symbol_data

# Suppress loud logs during evolutionary trials
logging.getLogger("optuna").setLevel(logging.INFO)

# Global Data Cache
_GLOBAL_DATA = None
_SYMBOLS = []
_DAYS = 7.0

def _get_data():
    global _GLOBAL_DATA
    if _GLOBAL_DATA is None:
        print(f"\n📥 [EVOLVER] Fetching historical data for {_SYMBOLS} over {_DAYS} days...")
        _GLOBAL_DATA = fetch_multi_symbol_data(_SYMBOLS, days=_DAYS)
    return _GLOBAL_DATA

def _clean_memory():
    """Limpia los Singletons del Engine para evitar OOM (Out Of Memory) en trials continuos."""
    import gc
    from core.omniscient_registry import registry
    from core.consensus_filter import _consensus_filter as consensus_filter
    from sophia.intelligence import SophiaIntelligence
    from utils.strategy_tracker import strategy_tracker
    from strategies.components.feature_engineering import FeatureEngineering
    import polars as pl
    import matplotlib.pyplot as plt
    
    if hasattr(registry, '_metrics'):
        registry._metrics.clear()
    if hasattr(registry, 'active_positions'):
        registry.active_positions.clear()
    if hasattr(consensus_filter, 'last_n_trades'):
        consensus_filter.last_n_trades.clear()
        
    if hasattr(SophiaIntelligence, '_instances'):
        SophiaIntelligence._instances.clear()
        
    if hasattr(strategy_tracker, 'trades'):
        strategy_tracker.trades.clear()
    if hasattr(strategy_tracker, 'all_time'):
        strategy_tracker.all_time.clear()
    if hasattr(strategy_tracker, 'by_symbol'):
        strategy_tracker.by_symbol.clear()
    if hasattr(strategy_tracker, 'by_horizon'):
        strategy_tracker.by_horizon.clear()

    if FeatureEngineering._instance is not None:
        if hasattr(FeatureEngineering._instance, '_result_cache'):
            FeatureEngineering._instance._result_cache.clear()
        if hasattr(FeatureEngineering._instance, 'metal_engines'):
            FeatureEngineering._instance.metal_engines.clear()
        if hasattr(FeatureEngineering._instance, 'feature_arenas'):
            FeatureEngineering._instance.feature_arenas.clear()
        FeatureEngineering._instance = None

    try:
        pl.toggle_string_cache(False)
    except:
        pass
    
    plt.close('all')
    
    # Forzar recolección profunda
    gc.collect()
    gc.collect()

def objective(trial):
    """
    Función Objetivo Holográfica: 
    Muta los parámetros centrales y evalúa todo el God Mode Engine 
    (RiskManager, Sophia, ML, Portfolio) simultáneamente.
    """
    # =====================================================================
    # 🧬 MUTACIÓN DEL ADN (Configuración Global)
    # =====================================================================
    
    # 1. RISK MANAGER
    Config.Risk.MAX_DRAWDOWN = trial.suggest_float("risk_max_drawdown", 0.05, 0.20, step=0.01)
    Config.Risk.PORTFOLIO_HEAT_LIMIT = trial.suggest_float("risk_portfolio_heat", 0.50, 0.90, step=0.05)
    Config.Risk.SOPHIA_CERTAINTY_FLOOR = trial.suggest_float("risk_sophia_floor", 0.60, 0.85, step=0.05)
    
    # 2. SCALPING ESTRATEGIA (Global Overrides)
    scalp_tp = trial.suggest_float("scalp_tp", 0.002, 0.020, step=0.001)
    scalp_sl = trial.suggest_float("scalp_sl", 0.005, 0.025, step=0.001)
    Config.Horizons.Scalping['tp_pct'] = scalp_tp
    Config.Horizons.Scalping['sl_pct'] = scalp_sl
    
    # 3. SWING ESTRATEGIA (Global Overrides)
    swing_tp = trial.suggest_float("swing_tp", 0.020, 0.150, step=0.010)
    swing_sl = trial.suggest_float("swing_sl", 0.010, 0.060, step=0.005)
    Config.Horizons.Swing['tp_pct'] = swing_tp
    Config.Horizons.Swing['sl_pct'] = swing_sl
    
    # 4. ENGINE & SOPHIA
    Config.Horizons.Scalping['cooldown_seconds'] = trial.suggest_int("engine_cooldown", 30, 300, step=30)
    Config.Horizons.Scalping['sophia_refit'] = trial.suggest_int("sophia_refit", 30, 150, step=10)
    Config.Horizons.Scalping['strength_threshold'] = trial.suggest_float("ml_strength_thresh", 0.50, 0.90, step=0.05)

    # 5. TRAILING STOPS (Dynamic Pursuit)
    # This edits the AdaptiveProfileEngine fallback mechanism indirectly, 
    # but for true integration we can override at Config.Trailing if implemented.
    trail_atr_mult = trial.suggest_float("trail_atr_mult", 0.5, 3.0, step=0.5)
    
    # Aplicar a todos los activos vía dictionary override en TP/SL
    for sym in _SYMBOLS:
        Config.Horizons.Scalping['tp_pct_per_asset'][sym] = scalp_tp
        Config.Horizons.Scalping['sl_pct_per_asset'][sym] = scalp_sl
        Config.Horizons.Swing['tp_pct_per_asset'][sym] = swing_tp
        Config.Horizons.Swing['sl_pct_per_asset'][sym] = swing_sl

    # =====================================================================
    # ⚡ EJECUCIÓN DEL GOD MODE ENGINE
    # =====================================================================
    all_data = _get_data()
    env_id = f"EVOLVER_{uuid.uuid4().hex[:6]}"
    
    try:
        results = run_global_backtest(
            all_data=all_data,
            symbols=_SYMBOLS,
            days=_DAYS,
            initial_capital=13.0, # Capital del usuario
            verbose=False,
            seed=42, # Para determinismo evolutivo
            scenario=env_id
        )
    except Exception as e:
        print(f"❌ Engine crash on Trial {trial.number}: {e}")
        return -1000.0
    finally:
        _clean_memory()
        
    if not results or "metrics" not in results:
        return -1000.0
        
    # =====================================================================
    # 🎯 FUNCIÓN OBJETIVO (TASA DE DOBLAMIENTO MULTIDIMENSIONAL)
    # =====================================================================
    metrics = results["metrics"]
    final_equity = metrics["Final Equity"]
    max_dd = metrics["Max Drawdown %"] / 100.0
    win_rate = metrics["Win Rate %"]
    trades = metrics["Total Trades"]
    
    # Castigos Severos
    if trades < 5 * len(_SYMBOLS): 
        # Insufficient market exposure
        raise optuna.TrialPruned("Insufficient trades")
        
    if final_equity < 13.0:
        # PnL Negativo
        return (final_equity - 13.0)
        
    if max_dd > 0.15:
        # Supera el 15% de DD (Inaceptable para 13 USD capital)
        raise optuna.TrialPruned(f"Drawdown too high: {max_dd*100:.1f}%")

    # Tasa de Doblamiento
    doublings = math.log2(final_equity / 13.0)
    td = doublings / _DAYS
    
    # Bono de Win Rate y Sharpe Estructural
    wr_bonus = (win_rate / 100.0) ** 2
    
    # Score Final: Favorece equidad pura, penalizada por DD, bonificada por WR
    safe_dd = max(max_dd, 0.001)
    fitness = td * (1.0 / safe_dd) * wr_bonus * math.log10(trades)
    
    print(f"🧬 Gen {trial.number:03d} | Equity: ${final_equity:.2f} | WR: {win_rate:.1f}% | DD: {max_dd*100:.1f}% | Score: {fitness:.2f}")
    return fitness

def run_omni_evolver(symbols: list, days: float, n_trials: int):
    global _SYMBOLS, _DAYS
    _SYMBOLS = symbols
    _DAYS = days
    
    print("\n" + "="*70)
    print("🌌 OMNI-EVOLVER SUPERMASSIVE UNIVERSAL RUNNER 🌌")
    print(f"🎯 Target Assets: {_SYMBOLS}")
    print(f"⏱️ Historic Horizon: {_DAYS} days")
    print(f"🧬 Generations/Trials: {n_trials}")
    print("="*70)
    
    _get_data()
    
    # SQLite persistente para guardar y resumir el estudio
    db_path = f"sqlite:///{os.path.join(_project_root, 'omniverse.db')}"
    study_name = f"OmniEvolver_{datetime.now().strftime('%Y%m%d')}"
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\n⚠️ Evolución Abortada por Usuario. Guardando estado en SQLite...")
    
    print("\n" + "="*70)
    print("🏆 EVOLUCIÓN COMPLETADA")
    print("="*70)
    
    if len(study.trials) == 0:
        print("No se completaron Trials.")
        return
        
    best = study.best_trial
    print(f"🏅 Mejor Puntuación: {best.value:.2f}")
    print("🧬 ADN Perfecto:")
    for key, value in best.params.items():
        print(f"    - {key}: {value}")
        
    # Guardar reporte
    import json
    report_path = os.path.join(_project_root, "data", "omni_evolver_best_dna.json")
    with open(report_path, "w") as f:
        json.dump(best.params, f, indent=4)
    print(f"\n💾 ADN Dorado Guardado en: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    parser.add_argument("--days", type=float, default=7.0, help="Days of historic data")
    parser.add_argument("--trials", type=int, default=100, help="Number of genetic mutations")
    args = parser.parse_args()
    
    syms = [s.strip().replace("USDT", "/USDT") if "USDT" in s and "/" not in s else s.strip() for s in args.symbols.split(",")]
    
    run_omni_evolver(syms, args.days, args.trials)
