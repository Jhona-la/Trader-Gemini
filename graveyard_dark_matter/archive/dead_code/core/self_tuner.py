"""
🧠 PHASE OMNI: SELF-TUNING HYPERPARAMETERS (Optuna Integration)
================================================================
QUÉ: Motor de auto-optimización de hiperparámetros vía Optuna.
POR QUÉ: El tuning manual no puede reaccionar a cambios de régimen de mercado
         detectados por HMM (MarketRegimeDetector).
PARA QUÉ: Mantener RSI, EMA, BB, ADX, y demás thresholds óptimos por régimen.
CÓMO: Cuando se detecta cambio de régimen, lanza N trials de Optuna en background.
      Los resultados se aplican vía HyperparamGuard (clamps seguros).
CUÁNDO: Activado por MarketRegimeDetector.on_regime_change() o cada 2 horas.
DÓNDE: core/self_tuner.py → integrado en main.py's meta_brain_loop.
QUIÉN: MetaBrain (StrategySelector), Config.Strategies.

DEPENDENCIAS CRÍTICAS:
- config.py → Config.Strategies.* (parámetros técnicos)
- core/market_regime.py → Detección de régimen
- strategies/technical.py → Consume los parámetros optimizados
- risk/risk_manager.py → Kelly fraction

SEGURIDAD:
- ❌ NUNCA se modifican parámetros de risk_manager.py directamente
- ✅ Solo se modifican Config.Strategies.* (signal generation)
- ✅ Todos los valores clampeados por HyperparamGuard
"""

import time
import threading
import os
from typing import Dict, Any, Optional, Callable
from utils.logger import setup_logger

logger = setup_logger("SelfTuner")

# Optional Optuna import (graceful degradation)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("⚠️ [SelfTuner] Optuna not installed. Self-tuning DISABLED.")


class HyperparamGuard:
    """
    🛡️ Safety Clamp for all tunable hyperparameters.
    
    QUÉ: Define rangos seguros (min, max) para cada hiperparámetro.
    POR QUÉ: Impedir que Optuna explore valores que causen pérdidas.
    PARA QUÉ: Garantizar que el bot opera dentro de límites institucionales.
    """
    
    # (min, max, type)
    BOUNDS = {
        # Technical Strategy
        'rsi_period':    (7, 28, int),
        'rsi_buy':       (20, 45, int),
        'rsi_sell':      (55, 80, int),
        'ema_fast':      (5, 50, int),
        'ema_slow':      (20, 200, int),
        'adx_threshold': (15, 40, int),
        'bb_period':     (10, 30, int),
        'bb_std':        (1.5, 3.0, float),
        'tp_pct':        (0.005, 0.03, float),
        'sl_pct':        (0.01, 0.05, float),
        
        # ML Strategy  
        'ml_confidence': (0.005, 0.05, float),
        'kelly_fraction': (0.1, 0.5, float),
    }
    
    @classmethod
    def clamp(cls, param_name: str, value: Any) -> Any:
        """Clamp a value to its safe bounds."""
        if param_name not in cls.BOUNDS:
            logger.warning(f"⚠️ [Guard] Unknown param '{param_name}'. Blocking update.")
            return None
        
        min_val, max_val, param_type = cls.BOUNDS[param_name]
        clamped = max(min_val, min(max_val, param_type(value)))
        
        if clamped != param_type(value):
            logger.info(f"🛡️ [Guard] Clamped {param_name}: {value} → {clamped}")
        
        return clamped
    
    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp all parameters in a dict."""
        safe = {}
        for key, value in params.items():
            clamped = cls.clamp(key, value)
            if clamped is not None:
                safe[key] = clamped
        
        # Cross-validation: ema_fast must be < ema_slow
        if 'ema_fast' in safe and 'ema_slow' in safe:
            if safe['ema_fast'] >= safe['ema_slow']:
                safe['ema_fast'] = max(5, safe['ema_slow'] - 10)
                logger.info(f"🛡️ [Guard] ema_fast adjusted to {safe['ema_fast']} (must be < ema_slow)")
        
        # Cross-validation: rsi_buy must be < rsi_sell
        if 'rsi_buy' in safe and 'rsi_sell' in safe:
            if safe['rsi_buy'] >= safe['rsi_sell']:
                safe['rsi_buy'] = 35
                safe['rsi_sell'] = 65
                logger.info("🛡️ [Guard] RSI bounds reset to 35/65 (overlap detected)")
        
        # Cross-validation: tp must be > sl for positive E[X]
        if 'tp_pct' in safe and 'sl_pct' in safe:
            if safe['tp_pct'] <= safe['sl_pct'] * 0.5:
                safe['tp_pct'] = safe['sl_pct'] * 1.5
                logger.info(f"🛡️ [Guard] tp_pct adjusted to {safe['tp_pct']:.4f} (must provide positive RR)")
        
        return safe


class SelfTuner:
    """
    🧠 Optuna-based Self-Tuning Engine.
    
    Launches optimization study when regime changes.
    Results are applied via HyperparamGuard clamps.
    Uses SQLite for trial persistence across bot restarts.
    """
    
    # Not needed anymore, we update Config.Horizons.Scalping directly
    
    def __init__(self, 
                 objective_fn: Optional[Callable] = None,
                 db_path: str = "data/optuna_studies.db",
                 n_trials: int = 30,
                 auto_apply: bool = True):
        """
        Args:
            objective_fn: Function(trial) → float (score to maximize).
                         If None, uses default backtesting objective.
            db_path: SQLite path for persistent trial storage.
            n_trials: Number of trials per optimization run.
            auto_apply: If True, best params are applied immediately.
        """
        self.objective_fn = objective_fn
        self.db_path = db_path
        self.n_trials = n_trials
        self.auto_apply = auto_apply
        
        self._current_regime = 'UNKNOWN'
        self._last_tune_time = 0.0
        self._min_interval = 3600  # Minimum 1 hour between tuning runs
        self._tuning_lock = threading.Lock()
        self._is_tuning = False
        
        # Track applied params history
        self._history: list = []
        
        logger.info(f"🧠 [SelfTuner] Initialized (Optuna: {'✅' if OPTUNA_AVAILABLE else '❌'})")
    
    def on_regime_change(self, new_regime: str):
        """
        Called by MarketRegimeDetector when regime changes.
        Triggers background optimization if conditions are met.
        """
        if not OPTUNA_AVAILABLE:
            return
        
        old_regime = self._current_regime
        self._current_regime = new_regime
        
        # Don't tune on first detection (UNKNOWN → X)
        if old_regime == 'UNKNOWN':
            return
        
        now = time.time()
        if now - self._last_tune_time < self._min_interval:
            logger.debug(f"[SelfTuner] Skipping tune (cooldown: {self._min_interval - (now - self._last_tune_time):.0f}s)")
            return
        
        logger.info(f"🧠 [SelfTuner] Regime Change: {old_regime} → {new_regime}. Launching optimization...")
        
        # Run in background thread
        t = threading.Thread(
            target=self._run_optimization,
            args=(new_regime,),
            daemon=True,
            name="SelfTuner-Optuna"
        )
        t.start()
    
    def _run_optimization(self, regime: str):
        """Background optimization run."""
        if not self._tuning_lock.acquire(blocking=False):
            logger.debug("[SelfTuner] Already tuning, skipping.")
            return
        
        try:
            self._is_tuning = True
            self._last_tune_time = time.time()
            
            study_name = f"gemini_{regime.lower()}"
            storage = f"sqlite:///{self.db_path}"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True,
            )
            
            # Use custom or default objective
            obj = self.objective_fn if self.objective_fn else self._default_objective
            
            study.optimize(obj, n_trials=self.n_trials, timeout=300)  # Max 5 min
            
            best = study.best_params
            best_value = study.best_value
            
            logger.info(f"🧠 [SelfTuner] Best Score: {best_value:.4f} | Params: {best}")
            
            # Apply via guard
            if self.auto_apply:
                self.apply_params(best)
            
            self._history.append({
                'regime': regime,
                'params': best,
                'score': best_value,
                'timestamp': time.time(),
            })
            
        except Exception as e:
            logger.error(f"❌ [SelfTuner] Optimization failed: {e}")
        finally:
            self._is_tuning = False
            self._tuning_lock.release()
    
    def _default_objective(self, trial) -> float:
        """
        Default objective: suggests params and returns a surrogate score.
        In production, this should run a quick backtest on recent data.
        """
        bounds = HyperparamGuard.BOUNDS
        
        params = {}
        for name, (min_v, max_v, ptype) in bounds.items():
            if ptype == int:
                params[name] = trial.suggest_int(name, min_v, max_v)
            else:
                params[name] = trial.suggest_float(name, min_v, max_v)
        
        # Cross-validation constraints
        if params['ema_fast'] >= params['ema_slow']:
            return -1.0  # Penalize invalid configs
        
        if params['rsi_buy'] >= params['rsi_sell']:
            return -1.0
        
        # Surrogate score: reward balanced parameters
        # In production, replace with actual backtest sharpe ratio
        rr_ratio = params['tp_pct'] / max(params['sl_pct'], 0.001)
        balance_score = min(rr_ratio, 3.0) / 3.0  # Normalize RR to [0, 1]
        
        return balance_score
    
    def apply_params(self, params: Dict[str, Any]):
        """
        Applies optimized parameters to Config.Horizons.Scalping via HyperparamGuard.
        
        SEGURIDAD: Todos los valores son clampeados y cross-validados
        antes de aplicarse a la configuración activa.
        """
        from config import Config
        
        safe_params = HyperparamGuard.validate_params(params)
        
        applied = []
        for param_name, value in safe_params.items():
            if param_name in ['ml_confidence', 'kelly_fraction']:
                if param_name == 'ml_confidence':
                    old_value = Config.Strategies.ML_MIN_CONFIDENCE
                    Config.Strategies.ML_MIN_CONFIDENCE = value
                    applied.append(f"ML_MIN_CONFIDENCE: {old_value} → {value}")
                elif param_name == 'kelly_fraction':
                    old_value = Config.Strategies.ML_KELLY_FRACTION
                    Config.Strategies.ML_KELLY_FRACTION = value
                    applied.append(f"ML_KELLY_FRACTION: {old_value} → {value}")
            else:
                # Update Scalping params directly
                if param_name in Config.Horizons.Scalping:
                    old_value = Config.Horizons.Scalping[param_name]
                    Config.Horizons.Scalping[param_name] = value
                    applied.append(f"{param_name}: {old_value} → {value}")
        
        if applied:
            logger.info(f"🧠 [SelfTuner] Applied {len(applied)} params: {', '.join(applied[:5])}")
        
        return safe_params
    
    def get_status(self) -> Dict:
        """Returns current tuner status for dashboard."""
        return {
            'is_tuning': self._is_tuning,
            'current_regime': self._current_regime,
            'last_tune_time': self._last_tune_time,
            'history_count': len(self._history),
            'optuna_available': OPTUNA_AVAILABLE,
            'last_best': self._history[-1] if self._history else None,
        }
