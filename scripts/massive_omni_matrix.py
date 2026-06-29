import os
import sys
import gc
import math
import uuid
import optuna
import logging
import argparse
import multiprocessing
from datetime import datetime

# Root paths
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from scripts.run_god_mode_backtest import run_global_backtest
from core.backtest_infra import fetch_multi_symbol_data

# Suppress loud logs
logging.getLogger("optuna").setLevel(logging.INFO)

# Global Data Cache for the Worker
_GLOBAL_DATA = None
_SYMBOLS = []
_DAYS = 7.0

def _get_data():
    global _GLOBAL_DATA
    if _GLOBAL_DATA is None:
        print(f"\n📥 [EVOLVER] Cargando histórico M.2 para {len(_SYMBOLS)} símbolos over {_DAYS} días...")
        _GLOBAL_DATA = fetch_multi_symbol_data(_SYMBOLS, days=_DAYS)
    return _GLOBAL_DATA

def _clean_memory():
    """Limpia la memoria del Worker para evitar fugas de RAM."""
    try:
        from core.omniscient_registry import registry
        from core.consensus_filter import _consensus_filter as consensus_filter
        from sophia.intelligence import SophiaIntelligence
        from utils.strategy_tracker import strategy_tracker
        from strategies.components.feature_engineering import FeatureEngineering
        import polars as pl
        import matplotlib.pyplot as plt
        
        if hasattr(registry, '_metrics'): registry._metrics.clear()
        if hasattr(registry, 'active_positions'): registry.active_positions.clear()
        if hasattr(consensus_filter, 'last_n_trades'): consensus_filter.last_n_trades.clear()
        if hasattr(SophiaIntelligence, '_instances'): SophiaIntelligence._instances.clear()
        
        if hasattr(strategy_tracker, 'trades'): strategy_tracker.trades.clear()
        if hasattr(strategy_tracker, 'all_time'): strategy_tracker.all_time.clear()
        if hasattr(strategy_tracker, 'by_symbol'): strategy_tracker.by_symbol.clear()
        if hasattr(strategy_tracker, 'by_horizon'): strategy_tracker.by_horizon.clear()

        if FeatureEngineering._instance is not None:
            if hasattr(FeatureEngineering._instance, '_result_cache'): FeatureEngineering._instance._result_cache.clear()
            if hasattr(FeatureEngineering._instance, 'metal_engines'): FeatureEngineering._instance.metal_engines.clear()
            if hasattr(FeatureEngineering._instance, 'feature_arenas'): FeatureEngineering._instance.feature_arenas.clear()
            FeatureEngineering._instance = None

        try:
            pl.toggle_string_cache(False)
            pl.clear_string_cache()
        except:
            pass
        
        plt.close('all')
    except Exception as e:
        pass
    
    # Garbage Collection Extrema
    import gc
    gc.collect()
    gc.collect()

def objective(trial):
    """
    Función Objetivo Cuántica: Evalúa todo el sistema en PARALELO.
    """
    try:
        # =====================================================================
        # 🧬 MUTACIÓN DEL ADN (Configuración Global)
        # =====================================================================
        
        # 1. RISK MANAGER
        Config.Risk.MAX_DRAWDOWN = trial.suggest_float("risk_max_drawdown", 0.05, 0.20, step=0.01)
        Config.Risk.PORTFOLIO_HEAT_LIMIT = trial.suggest_float("risk_portfolio_heat", 0.50, 0.90, step=0.05)
        Config.Risk.SOPHIA_CERTAINTY_FLOOR = trial.suggest_float("risk_sophia_floor", 0.50, 0.85, step=0.05)
        
        # 2. SCALPING ESTRATEGIA
        scalp_tp = trial.suggest_float("scalp_tp", 0.002, 0.015, step=0.001)
        scalp_sl = trial.suggest_float("scalp_sl", 0.002, 0.015, step=0.001)
        Config.Horizons.Scalping['tp_pct'] = scalp_tp
        Config.Horizons.Scalping['sl_pct'] = scalp_sl
        
        # 3. SWING ESTRATEGIA
        swing_tp = trial.suggest_float("swing_tp", 0.020, 0.150, step=0.010)
        swing_sl = trial.suggest_float("swing_sl", 0.010, 0.080, step=0.005)
        Config.Horizons.Swing['tp_pct'] = swing_tp
        Config.Horizons.Swing['sl_pct'] = swing_sl
        
        # 4. ENGINE & SOPHIA
        Config.Horizons.Scalping['cooldown_seconds'] = trial.suggest_int("engine_cooldown", 15, 300, step=15)
        Config.Horizons.Scalping['sophia_refit'] = trial.suggest_int("sophia_refit", 30, 240, step=30)
        Config.Horizons.Scalping['strength_threshold'] = trial.suggest_float("ml_strength_thresh", 0.50, 0.90, step=0.05)

        trail_atr_mult = trial.suggest_float("trail_atr_mult", 0.5, 3.0, step=0.5)
        
        # Override per asset
        for sym in _SYMBOLS:
            Config.Horizons.Scalping['tp_pct_per_asset'][sym] = scalp_tp
            Config.Horizons.Scalping['sl_pct_per_asset'][sym] = scalp_sl
            Config.Horizons.Swing['tp_pct_per_asset'][sym] = swing_tp
            Config.Horizons.Swing['sl_pct_per_asset'][sym] = swing_sl

        # =====================================================================
        # ⚡ EJECUCIÓN DEL GOD MODE ENGINE
        # =====================================================================
        all_data = _get_data()
        env_id = f"M_MATRIX_{uuid.uuid4().hex[:6]}"
        
        # Ejecuta el simulador completo (todas las estrategias interactuando)
        results = run_global_backtest(
            all_data=all_data,
            symbols=_SYMBOLS,
            days=_DAYS,
            initial_capital=13.0, # Capital ultra estricto
            verbose=False,
            seed=42,
            scenario=env_id
        )
        
        score = results.get('score', -1000.0)
        
        # Castigo drástico por quemar la cuenta
        if results.get('max_drawdown', 1.0) > 0.5:
            score -= 1000.0
            
        return score

    except Exception as e:
        print(f"❌ [WORKER CRASH] Trial {trial.number} failed: {e}")
        return -1000.0
    finally:
        _clean_memory()


def start_worker(db_path, study_name, n_trials):
    """Inicializa un worker aislado para evitar compartición de estado global"""
    study = optuna.load_study(study_name=study_name, storage=db_path)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

def main():
    parser = argparse.ArgumentParser(description="Quantum Omni-Evolver Matrix")
    parser.add_argument('--symbols', type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,DOGE/USDT,XRP/USDT,ADA/USDT,AVAX/USDT,LINK/USDT,LTC/USDT,MATIC/USDT,DOT/USDT,UNI/USDT,ATOM/USDT,NEAR/USDT,AAVE/USDT,ALGO/USDT,ICP/USDT,APT/USDT,ARB/USDT", help="Lista de símbolos separados por comas")
    parser.add_argument('--days', type=float, default=7.0, help="Días de backtest")
    parser.add_argument('--trials', type=int, default=1000, help="Número de simulaciones por worker")
    parser.add_argument('--workers', type=int, default=4, help="Número de procesos concurrentes (max cores)")
    args = parser.parse_args()

    global _SYMBOLS, _DAYS
    _SYMBOLS = [s.strip() for s in args.symbols.split(',') if s.strip()]
    _DAYS = args.days

    # Preparar el Storage
    os.makedirs(os.path.join(_project_root, 'data'), exist_ok=True)
    db_path = f"sqlite:///{os.path.join(_project_root, 'data', 'omniverse_massive.db')}"
    study_name = f"MASSIVE_EVOLUTION_MATRIX_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"============================================================")
    print(f"🌌 INICIANDO QUANTUM OMNI-EVOLVER MATRIX (DISTRIBUTED)")
    print(f"============================================================")
    print(f"Símbolos Evaluados simultáneamente: {len(_SYMBOLS)}")
    print(f"Días Históricos por Simulación: {_DAYS}")
    print(f"Workers Paralelos (CPU Cores): {args.workers}")
    print(f"Trials por Worker: {args.trials} (Total {args.trials * args.workers})")
    print(f"Almacenamiento: {db_path}")
    print(f"============================================================")

    # Crear el estudio centralizado
    study = optuna.create_study(
        study_name=study_name, 
        storage=db_path, 
        direction="maximize",
        load_if_exists=True
    )
    
    # Precargar data en el proceso principal para que sea cacheable o descargada 1 sola vez
    _get_data()

    # Lanzar los workers en paralelo
    processes = []
    for i in range(args.workers):
        p = multiprocessing.Process(target=start_worker, args=(db_path, study_name, args.trials))
        p.start()
        processes.append(p)
        
    print("🚀 Workers distribuidos lanzados exitosamente.")
    
    # Esperar finalización
    for p in processes:
        p.join()
        
    print(f"============================================================")
    print(f"🏆 MATRIZ EVOLUTIVA COMPLETADA")
    print(f"============================================================")
    
    best = study.best_trial
    print(f"🏅 Mejor Puntuación General: {best.value:.2f}")
    print(f"🧬 ADN Dorado Extraído:")
    for key, value in best.params.items():
        print(f"    - {key}: {value}")
        
    import json
    best_dna_path = os.path.join(_project_root, 'data', 'massive_best_dna.json')
    with open(best_dna_path, 'w') as f:
        json.dump(best.params, f, indent=4)
        
    print(f"\n💾 ADN Dorado Guardado en: {best_dna_path}")

if __name__ == "__main__":
    main()
