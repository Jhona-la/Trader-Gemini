import sys
import os
import time
import json
import uuid
import logging
from datetime import datetime
from copy import deepcopy
import gc

import optuna
import pandas as pd
import numpy as np

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
import importlib.util

# Cargar adaptive_config.py
adaptive_config_path = os.path.join(_project_root, "config", "adaptive_config.py")
spec = importlib.util.spec_from_file_location("adaptive_config", adaptive_config_path)
adaptive_config_module = importlib.util.module_from_spec(spec)
sys.modules["adaptive_config"] = adaptive_config_module
spec.loader.exec_module(adaptive_config_module)
adaptive_config = adaptive_config_module.adaptive_config

from core.omni_fitness import calculate_omni_fitness
from core.backtest_infra import fetch_multi_symbol_data
from core.engine import Engine
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from core.strategy_registry import UniversalStrategyRegistry

from utils.logger import logger

def update_config_dict(target_obj, overrides_dict):
    """Recursively update config matrix."""
    for key, value in overrides_dict.items():
        if isinstance(value, dict) and key in target_obj and isinstance(target_obj[key], dict):
            update_config_dict(target_obj[key], value)
        else:
            target_obj[key] = value

def generate_blueprint_overrides(trial):
    """
    Blueprint Search Space: 
    Profundamente muta parámetros de RiskManager, Sniper, Technical y Pattern
    para los 3 horizontes.
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
                    'capital_allocation_base_pct': trial.suggest_float('micro_alloc', 0.1, 0.5, step=0.05)
                },
                'por_activo': {
                    'BTC': {
                        'tp_pct_default': trial.suggest_float('micro_tp', 0.05, 0.5, step=0.01),
                        'sl_pct_default': trial.suggest_float('micro_sl', 0.05, 0.25, step=0.01),
                        'leverage': trial.suggest_categorical('micro_lev', [10, 15, 20])
                    }
                }
            },
            'SCALP': {
                'global_horizon': {
                    'capital_allocation_base_pct': trial.suggest_float('scalp_alloc', 0.2, 0.6, step=0.05)
                },
                'por_activo': {
                    'BTC': {
                        'tp_pct_default': trial.suggest_float('scalp_tp', 0.3, 1.0, step=0.05),
                        'sl_pct_default': trial.suggest_float('scalp_sl', 0.15, 0.6, step=0.05),
                        'leverage': trial.suggest_categorical('scalp_lev', [5, 10, 15])
                    }
                }
            },
            'SWING': {
                'global_horizon': {
                    'capital_allocation_base_pct': trial.suggest_float('swing_alloc', 0.1, 0.5, step=0.05)
                },
                'por_activo': {
                    'BTC': {
                        'tp_pct_default': trial.suggest_float('swing_tp', 1.5, 4.0, step=0.2),
                        'sl_pct_default': trial.suggest_float('swing_sl', 0.8, 2.0, step=0.1),
                        'leverage': trial.suggest_categorical('swing_lev', [2, 3, 5])
                    }
                }
            }
        }
    }

def simulate_omni_backtest(all_data, symbols, days, initial_capital, blueprint):
    """
    Simulates MICROSCALPING, SCALPING, and SWING simultaneously.
    """
    # Initialize Portfolio
    portfolio = Portfolio(initial_capital=initial_capital, auto_save=False)
    
    # Initialize RiskManager with mutated Blueprint limits
    risk_manager = RiskManager(max_concurrent_positions=10, portfolio=portfolio)
    if hasattr(risk_manager.cvar_calc, 'confidence_level'):
        risk_manager.cvar_calc.confidence_level = blueprint['BLUEPRINT_RISK']['cvar_confidence']
    Config.Risk.MAX_DRAWDOWN = blueprint['BLUEPRINT_RISK']['daily_drawdown_limit'] * 100
    
    # Inject OmniScore weights into global Config for consensus filter
    if not hasattr(Config, 'OmniScore'):
        Config.OmniScore = type('OmniScore', (), {})
    if 'BLUEPRINT_OMNISCORE' in blueprint:
        omni = blueprint['BLUEPRINT_OMNISCORE']
        Config.OmniScore.w_ml = omni['w_ml']
        Config.OmniScore.w_technical = omni['w_technical']
        Config.OmniScore.w_phalanx = omni['w_phalanx']
        Config.OmniScore.w_statarb = omni['w_statarb']
        Config.OmniScore.master_threshold = omni['master_threshold']
        Config.OmniScore.ml_threshold_bull = 0.55
        Config.OmniScore.ml_threshold_bear = 0.55
        
    events_queue = []
    
    # Fake a data provider for the backtest
    class DummyDataProvider:
        def __init__(self):
            self.current_time_ms = 0
    data_provider = DummyDataProvider()

    global_dependencies = {
        'data_provider': data_provider,
        'events_queue': events_queue,
        'portfolio': portfolio,
        'executor': None, # No real executor needed
        'risk_manager': risk_manager,
        'sentiment_loader': None,
        'models_dir': os.path.join(_project_root, 'models'),
        'db_path': os.path.join(_project_root, 'data', 'backtest_blueprint.db'),
    }

    all_strategies = []
    
    # 1. Instantiate ALL horizons simultaneously
    horizons = ['MICROSCALPING', 'SCALPING', 'SWING']
    for h in horizons:
        global_dependencies['horizon'] = h
        strats = UniversalStrategyRegistry.create_all(**global_dependencies)
        
        # Inject Blueprint parameters directly into the instantiated strategy objects
        for s in strats:
            s_name = str(s.__class__.__name__).upper()
            if 'SNIPER' in s_name:
                s.volume_spike_multiplier = blueprint['BLUEPRINT_SNIPER']['volume_spike_multiplier']
                s.absorption_threshold = blueprint['BLUEPRINT_SNIPER']['absorption_threshold_pct']
            elif 'TECHNICAL' in s_name:
                s.rsi_oversold = blueprint['BLUEPRINT_TECHNICAL']['rsi_oversold']
                s.rsi_overbought = blueprint['BLUEPRINT_TECHNICAL']['rsi_overbought']
            elif 'PATTERN' in s_name:
                s.wick_filter_strictness = blueprint['BLUEPRINT_PATTERN']['wick_filter_strictness']
                
        all_strategies.extend(strats)
        
    engine = Engine(events_queue=events_queue)
    engine.register_portfolio(portfolio)
    engine.register_risk_manager(risk_manager)
    engine.register_data_handler(data_provider)
    engine.register_execution_handler(None)
    
    # Map strategies manually into the engine (simplified for backtest mock)
    engine.strategies = {sym: [] for sym in symbols}
    engine.global_strategies = []
    for s in all_strategies:
        if hasattr(s, 'symbol') and s.symbol in engine.strategies:
            engine.strategies[s.symbol].append(s)
        else:
            engine.global_strategies.append(s)

    # Convert all_data to chronological events (Simplified O(N) linear pass)
    events_timeline = []
    for sym, intervals in all_data.items():
        if '1m' not in intervals: continue
        bars = intervals['1m']
        # Convert structured array or list of dicts to flat events
        if isinstance(bars, list):
            for b in bars:
                events_timeline.append((b['timestamp'], sym, b))
        elif isinstance(bars, np.ndarray) and hasattr(bars.dtype, 'names'):
            for i in range(len(bars)):
                b = {
                    'timestamp': int(bars['timestamp'][i]),
                    'close': float(bars['close'][i]),
                    'high': float(bars['high'][i]),
                    'low': float(bars['low'][i]),
                    'volume': float(bars['volume'][i])
                }
                events_timeline.append((b['timestamp'], sym, b))

    # Sort chronologically
    events_timeline.sort(key=lambda x: x[0])
    
    if not events_timeline:
        return {'metrics': {'win_rate': 0.0, 'total_trades': 0, 'max_drawdown': 1.0}, 'trades': []}

    # Play timeline
    from core.events import MarketEvent
    trades_executed = []
    
    for ts, sym, bar in events_timeline:
        data_provider.current_time_ms = ts
        
        # Update Portfolio virtual prices
        for v_key, pos in portfolio.virtual_ledger.items():
            if v_key.startswith(sym):
                pos['current_price'] = bar['close']
                
        me = MarketEvent(
            symbol=sym,
            close_price=bar['close'],
            high_price=bar['high'],
            low_price=bar['low'],
            is_closed=True
        )
        engine._process_market_event(me)
        
        # Process resulting signals
        while events_queue:
            ev = events_queue.pop(0)
            if ev.type == "SIGNAL":
                # Mock execution: accept signal instantly
                if ev.signal_type.name in ['LONG', 'SHORT']:
                    # Size via Portfolio
                    direction = "LONG" if ev.signal_type.name == 'LONG' else "SHORT"
                    avail_cash = portfolio.get_available_cash(ev.horizon)
                    if avail_cash > 5.0:
                        qty = (avail_cash * 0.95) / bar['close'] # Fake size
                        # Open
                        v_key = f"{sym}_{ev.horizon}_{direction}"
                        portfolio.virtual_ledger[v_key] = {
                            'quantity': qty if direction == 'LONG' else -qty,
                            'avg_price': bar['close'],
                            'current_price': bar['close'],
                            'sl_pct': ev.stop_loss_pct,
                            'tp_pct': ev.take_profit_pct,
                            'horizon': ev.horizon,
                            'strategy_id': ev.strategy_id,
                            'entry_time': ts
                        }
                        portfolio.used_margin += avail_cash * 0.95
                elif ev.signal_type.name == 'EXIT':
                    direction = "LONG" if "LONG" in ev.reason else "SHORT"
                    v_key = f"{sym}_{ev.horizon}_{direction}"
                    if v_key in portfolio.virtual_ledger:
                        pos = portfolio.virtual_ledger[v_key]
                        qty = pos['quantity']
                        avg = pos['avg_price']
                        curr = bar['close']
                        pnl = (curr - avg) * qty
                        pnl_pct = (curr - avg) / avg * (1 if direction == 'LONG' else -1)
                        trades_executed.append({
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl,
                            'horizon': ev.horizon
                        })
                        portfolio.current_cash += pnl
                        portfolio.used_margin = max(0, portfolio.used_margin - (abs(qty)*avg))
                        del portfolio.virtual_ledger[v_key]

    # Force close remaining
    for v_key, pos in portfolio.virtual_ledger.items():
        qty = pos['quantity']
        avg = pos['avg_price']
        curr = pos['current_price']
        pnl = (curr - avg) * qty
        direction = 'LONG' if qty > 0 else 'SHORT'
        pnl_pct = (curr - avg) / avg * (1 if direction == 'LONG' else -1)
        trades_executed.append({
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl,
            'horizon': pos['horizon']
        })
        
    wins = len([t for t in trades_executed if t['pnl_pct'] > 0])
    total = len(trades_executed)
    win_rate = (wins/total*100) if total > 0 else 0.0
    
    return {
        'metrics': {
            'win_rate': win_rate,
            'total_trades': total,
            'max_drawdown': 0.0, # Placeholder
        },
        'trades': trades_executed
    }


def objective(trial, all_data, symbols, days, original_matrix):
    overrides = generate_blueprint_overrides(trial)
    
    # Normalizar asignaciones
    m_alloc = overrides['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct']
    s_alloc = overrides['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct']
    sw_alloc = overrides['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct']
    total_alloc = m_alloc + s_alloc + sw_alloc
    if total_alloc > 1.0:
        overrides['MATRIX_OVERRIDES']['MICRO']['global_horizon']['capital_allocation_base_pct'] /= total_alloc
        overrides['MATRIX_OVERRIDES']['SCALP']['global_horizon']['capital_allocation_base_pct'] /= total_alloc
        overrides['MATRIX_OVERRIDES']['SWING']['global_horizon']['capital_allocation_base_pct'] /= total_alloc
        
    adaptive_config.matrix = deepcopy(original_matrix)
    update_config_dict(adaptive_config.matrix, overrides['MATRIX_OVERRIDES'])
    
    try:
        results = simulate_omni_backtest(all_data, symbols, days, 13.0, overrides)
    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return -9999.0
        
    trades = results['trades']
    if len(trades) < 5:
        return -9999.0
        
    pnls = np.array([t['pnl_pct']/100.0 for t in trades])
    win_rate = results['metrics']['win_rate']
    
    score = calculate_omni_fitness(pnls, win_rate, 0.1, len(trades), 13.0)
    
    gc.collect()
    return score

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,SOL/USDT")
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    logger.info("="*60)
    logger.info("🧠 BLUEPRINT OMNI-EVOLVER INICIANDO...")
    logger.info("="*60)
    
    all_data = fetch_multi_symbol_data(symbols, args.days)
    original_matrix = deepcopy(adaptive_config.matrix)
    
    study_name = f"blueprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda t: objective(t, all_data, symbols, args.days, original_matrix), n_trials=args.trials)
    
    best_trial = study.best_trial
    logger.info("🏆 MEJOR BLUEPRINT ENCONTRADO")
    logger.info(f"Score: {best_trial.value}")
    
    # Generate full blueprint
    blueprint = generate_blueprint_overrides(best_trial)
    
    output_path = os.path.join(_project_root, "data", f"blueprint_master_{study_name}.json")
    with open(output_path, "w") as f:
        json.dump(blueprint, f, indent=4)
        
    logger.info(f"💾 Guardado en {output_path}")

if __name__ == "__main__":
    main()
