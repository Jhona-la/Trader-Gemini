import os
import sys
import math
import optuna
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from scripts.run_god_mode_backtest import run_global_backtest
from core.backtest_infra import fetch_binance_data

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.INFO)

# Global data cache to avoid re-fetching
_GLOBAL_DATA = None
_SYMBOLS = ["BTC/USDT"]
_DAYS = 4

def get_data():
    global _GLOBAL_DATA
    if _GLOBAL_DATA is None:
        print("📥 Fetching historical data for Optuna...")
        all_data = {}
        for sym in _SYMBOLS:
            df = fetch_binance_data(sym, days=_DAYS)
            if df is not None and not df.empty:
                all_data[sym] = df
        _GLOBAL_DATA = all_data
    return _GLOBAL_DATA

def objective(trial):
    """
    Optuna Objective Function: Maximize Tasa de Doblamiento (TD).
    """
    # 1. Mutate Config dynamically based on trial suggestions
    # We target the Swing and Scalping TP/SL which influences `dynamic_b`
    
    # Suggest TP and SL for Scalping
    scalp_tp = trial.suggest_float("scalp_tp", 0.005, 0.05, step=0.005)
    scalp_sl = trial.suggest_float("scalp_sl", 0.005, 0.03, step=0.005)
    
    # Suggest TP and SL for Swing
    swing_tp = trial.suggest_float("swing_tp", 0.02, 0.15, step=0.01)
    swing_sl = trial.suggest_float("swing_sl", 0.01, 0.05, step=0.01)

    # Apply to Config (Horizons properties are dictionaries)
    Config.Horizons.Scalping['tp_pct'] = scalp_tp
    Config.Horizons.Scalping['sl_pct'] = scalp_sl
    Config.Horizons.Swing['tp_pct'] = swing_tp
    Config.Horizons.Swing['sl_pct'] = swing_sl

    # 2. Run the God Mode Backtest silently
    all_data = get_data()
    
    # FASE IV: Aislar ID de entorno
    import uuid
    import gc
    env_id = f"OPTUNA_{uuid.uuid4().hex[:6]}"
    
    try:
        results = run_global_backtest(
            all_data=all_data,
            symbols=_SYMBOLS,
            days=_DAYS,
            initial_capital=13.0,
            verbose=False,
            seed=42, # Deterministic evaluation
            scenario=env_id
        )
    except Exception as e:
        import traceback
        print(f"Engine crash: {e}")
        traceback.print_exc()
        return -10000.0
    finally:
        # FASE IV: Determinismo y Aislamiento de Memoria
        # Limpiar Singletons
        from core.omniscient_registry import registry
        from core.consensus_filter import _consensus_filter as consensus_filter
        if hasattr(registry, '_metrics'):
            registry._metrics.clear()
        if hasattr(registry, 'active_positions'):
            registry.active_positions.clear()
        if hasattr(consensus_filter, 'last_n_trades'):
            consensus_filter.last_n_trades.clear()
        # Forzar recolección de basura para no desbordar RAM en 50 trials
        gc.collect()

    # 3. Extract metrics
    if not results or "metrics" not in results:
        return -10000.0
        
    metrics = results["metrics"]
    final_capital = metrics.get("Final Equity", 13.0)
    max_dd_pct = metrics.get("Max Drawdown %", 0.0) / 100.0
    total_trades = metrics.get("Total Trades", 0)
    sharpe = metrics.get("Sharpe Ratio", 0.0)
    pnl_bruto = final_capital - 13.0

    if total_trades < 5:
        # Not enough statistical significance
        raise optuna.TrialPruned("Not enough trades (Pruned)")

    # 4. Calculate TD and Penalty
    if final_capital <= 0.1:
        doublings = -10.0
    else:
        doublings = math.log2(final_capital / 13.0)
        
    td = doublings / _DAYS

    # FASE III: Función Objetivo Holográfica
    safe_dd = max(max_dd_pct, 0.001)  # Avoid div by zero
    
    # Penalidad Estricta
    if sharpe < 1.0 or max_dd_pct > 0.20:
        penalty = -1000.0
    else:
        penalty = 0.0

    # Score: TD * (Eficiencia) * Retorno Nominal
    # Usamos log(total_trades) como estabilizador
    score = td * (sharpe / safe_dd) * pnl_bruto * math.log(total_trades + 1)
    
    final_score = score + penalty
    
    return final_score

def run_optimization(n_trials=5):
    print(f"🚀 Iniciando Optuna TD Optimizer ({n_trials} trials)...")
    
    # Ensure data is cached before starting Optuna to measure accurately
    get_data()
    
    study = optuna.create_study(direction="maximize", study_name="Tasa_Doblamiento_PhaseVII")
    study.optimize(objective, n_trials=n_trials, n_jobs=1) # n_jobs=1 because of global Config mutations
    
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE")
    print("="*50)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("⚠️ No trials completed successfully (all were pruned or failed).")
    else:
        print(f"Best TD Score: {study.best_value:.4f}")
        print("Best Params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
    print("="*50)

if __name__ == "__main__":
    trials = 5
    if len(sys.argv) > 1:
        trials = int(sys.argv[1])
    run_optimization(trials)
