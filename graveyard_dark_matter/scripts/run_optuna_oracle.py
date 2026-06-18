"""
🔮 PHASE 3: OPTUNA CONTINUOUS ORACLE (EVOLUTIONARY TUNING)
==========================================================
QUÉ: Optimizador Bayesiano para los parámetros de la estrategia y Meta-Labeling.
POR QUÉ: Para alcanzar consistentemente un Win Rate >70%, el ML necesita thresholds
         y etiquetas de entrenamiento adaptadas al régimen, no heurísticas fijas.
PARA QUÉ: Optimizar iterativamente (tp_mult, sl_mult, ml_confidence) simulando el
          motor de inferencia XGBoost con las métricas de Microestructura inyectadas.
CÓMO: TPESampler de Optuna. Mutando en crudo el diccionario HORIZON_PROFILES
      de `run_multi_horizon_backtest` en memoria antes de evaluar el trail.
CUÁNDO: Ejecutado periódicamente (diario/semanal) o al detectar cambios fuertes HMM.
DÓNDE: scripts/run_optuna_oracle.py
"""
import os
import sys
import json
import logging
from typing import Dict, Any

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Ensure Trader Gemini root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_multi_horizon_backtest as runner

# Setup Logging
optuna.logging.set_verbosity(optuna.logging.INFO)
logger = logging.getLogger("OptunaOracle")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class OptunaOracle:
    def __init__(self, symbol: str = 'BTC/USDT', horizon_days: int = 1, db_path: str = "sqlite:///data/optuna_studies.db"):
        self.symbol = symbol
        self.horizon_days = horizon_days
        self.db_path = db_path
        
        # Load data once to speed up all trials
        logger.info(f"📡 Fetching data for {symbol} ({horizon_days}D Horizon)...")
        fetch_days = horizon_days + (8 if horizon_days <= 1 else 15)
        self.df = runner.fetch_data(symbol, fetch_days)
        if self.df is None or len(self.df) < 500:
            raise ValueError(f"Insufficient data for {symbol}")
            
        logger.info(f"✅ Data cached: {len(self.df)} candles.")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna TPE.
        Optimizes exclusively for the 'ML_XGBoost' Strategy.
        Target: Maximize Win Rate (must be > 70%) combined with Sharpe Ratio.
        """
        # 1. Parameter Sugestion based on Bayesian distribution
        sl_mult = trial.suggest_float("sl_mult", 1.5, 6.0, step=0.5)
        tp_mult = trial.suggest_float("tp_mult", 2.0, 10.0, step=0.5)
        
        # XGBoost Veto Thresholds
        ml_confidence = trial.suggest_float("ml_confidence", 0.55, 0.85, step=0.01)
        ml_lookahead = trial.suggest_int("ml_lookahead", 30, 240, step=30)
        
        # Constraint: TP > SL + 1 to maintain positive expectancy
        if tp_mult <= sl_mult * 1.2:
            raise optuna.exceptions.TrialPruned("Negative / 1:1 Risk Reward Ratio.")

        # 2. Inject into Runner context (In-Memory Override)
        # This replaces the hardcoded values just for this simulation trial
        runner.HORIZON_PROFILES[self.horizon_days]['sl_mult'] = sl_mult
        runner.HORIZON_PROFILES[self.horizon_days]['tp_mult'] = tp_mult
        runner.HORIZON_PROFILES[self.horizon_days]['ml_confidence'] = ml_confidence
        runner.HORIZON_PROFILES[self.horizon_days]['ml_lookahead'] = ml_lookahead
        
        # Force strict retraining dynamically based on horizon
        if self.horizon_days == 1:
            runner.HORIZON_PROFILES[self.horizon_days]['ml_retrain'] = trial.suggest_int("ml_retrain", 120, 720, step=60)
            
        # 3. Execute the Backtest for ML_XGBoost only
        try:
            result = runner.run_strategy_backtest(
                df=self.df,
                symbol=self.symbol,
                strategy_name='ML_XGBoost',
                initial_capital=1000.0,
                leverage=1,
                horizon_days=self.horizon_days
            )
        except Exception as e:
            logger.error(f"Trial failed during execution: {e}")
            return -999.0
            
        # 4. Extract Metrics
        win_rate = result.get('win_rate', 0.0)
        sharpe = result.get('sharpe', 0.0)
        trades = result.get('trades', 0)
        pnl = result.get('pnl_pct', 0.0)
        max_dd = result.get('max_drawdown', 100.0)

        # 5. Fitness Calculation 
        if trades < 5:
            # Penalize inactivity
            return -100.0
            
        if pnl < 0:
            # Penalize loss proportionally to the drawdown
            return -50.0 - max_dd

        # Base score is Sharpe Ratio
        fitness = sharpe 
        
        # Primary Multiplier: Did it breach the >70% Win Rate institutional goal?
        if win_rate > 70.0:
            fitness *= 2.0  # Massive reward structural safety
        elif win_rate < 50.0:
            fitness *= 0.5  # Heavy penalty for coin-flip dynamics
            
        return float(fitness)

    def optimize(self, n_trials: int = 50):
        """Runs the optimization study."""
        study_name = f"oracle_{self.symbol.replace('/', '')}_{self.horizon_days}D"
        
        os.makedirs("data", exist_ok=True)
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.db_path,
            direction="maximize",
            sampler=TPESampler(multivariate=True, seed=42),
            pruner=MedianPruner(n_warmup_steps=10),
            load_if_exists=True
        )
        
        logger.info(f"🚀 Starting Optuna Oracle for {self.symbol} [{self.horizon_days}D]")
        study.optimize(self.objective, n_trials=n_trials)
        
        best = study.best_trial
        logger.info(f"🏆 BEST RESULTS (Trial #{best.number}):")
        logger.info(f"  Fitness: {best.value:.4f}")
        logger.info(f"  Params: {json.dumps(best.params, indent=2)}")
        
        # Save to JSON profile
        profile_path = f"data/oracle_profile_{self.horizon_days}D.json"
        with open(profile_path, 'w') as f:
            json.dump(best.params, f, indent=2)
        logger.info(f"📁 Optimal profile saved to {profile_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 7, 15, 30], help="Horizon days")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    
    args = parser.parse_args()
    
    oracle = OptunaOracle(symbol=args.symbol, horizon_days=args.horizon)
    oracle.optimize(n_trials=args.trials)
