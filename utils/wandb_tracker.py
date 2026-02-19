"""
OMEGA PROTOCOL: WandB Experiment Tracker (Phase 99/7-8)
=======================================================
QUÉ: Wrapper para Weights & Biases que sube métricas de evolución,
     RL learning curves, y comparativas entre épocas a la nube.
POR QUÉ: Visualización cloud de la convergencia del Algoritmo Genético
         y del Reinforcement Learning permite debugging a distancia.
PARA QUÉ: (1) Ver gráficas de fitness vs generación,
           (2) Comparar rendimiento entre épocas,
           (3) Detectar overfitting o divergencia tempranamente.
CÓMO: wandb.init() → wandb.log() → wandb.finish(). Async HTTP.
CUÁNDO: Llamado desde ShadowDarwin.run_epoch() y ml_strategy.
DÓNDE: utils/wandb_tracker.py
QUIÉN: Singleton `wandb_tracker` importado donde se necesite.

NOTA: WandB es OPCIONAL — si no hay API key, se loguea localmente.
"""
import os
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("WandBTracker")

# Try to import wandb — graceful degradation if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.info("📊 WandB not installed — experiment tracking will be local-only.")


class WandBTracker:
    """
    Manages WandB experiment runs for the Trader Gemini ecosystem.
    
    Usage:
        tracker = WandBTracker()
        tracker.init_run(project="trader-gemini", config={...})
        tracker.log_generation(gen_id=1, fitness=0.85, ...)
        tracker.finish()
    """
    
    def __init__(self):
        self._run = None
        self._active = False
        self._local_log = []  # Fallback when WandB not available
        self._run_name = None
    
    @property
    def is_active(self) -> bool:
        return self._active and WANDB_AVAILABLE and self._run is not None
    
    def init_run(self, project: str = "trader-gemini", 
                 entity: str = "jhonala-none",
                 config: Optional[Dict] = None,
                 name: Optional[str] = None,
                 tags: Optional[list] = None,
                 resume: str = "allow") -> bool:
        """
        Initialize a WandB run.
        
        Args:
            project: WandB project name
            config: Hyperparameters and config to track
            name: Run name (auto-generated if None)
            tags: Tags for filtering
            resume: 'allow' | 'must' | 'never'
            
        Returns:
            True if WandB initialized, False if local fallback
        """
        if not WANDB_AVAILABLE:
            logger.info("📊 WandB not available — using local fallback.")
            self._run_name = name or f"local_{int(time.time())}"
            return False
        
        # Phase 99: WandB will automatically use the key from 'wandb login' or environment variables.
        # We manually set it if the user provided WandB_Key in .env to ensure compatibility.
        api_key = os.getenv("WANDB_API_KEY") or os.getenv("WandB_Key")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        
        try:
            self._run_name = name or f"omega_{int(time.time())}"
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=self._run_name,
                config=config or {},
                tags=tags or ["omega", "trader-gemini"],
                resume=resume,
                reinit=True
            )
            self._active = True
            logger.info(f"📊 WandB Run initialized: {self._run_name} ({project})")
            return True
            
        except Exception as e:
            logger.error(f"❌ WandB init failed: {e}")
            self._active = False
            return False
    
    def log_generation(self, gen_id: int, fitness: float, 
                       diversity: float = 0.0,
                       params: Optional[Dict] = None,
                       symbol: str = "unknown",
                       method: str = "genetic"):
        """
        Log a single generation/trial result.
        
        Args:
            gen_id: Generation or trial number
            fitness: Fitness score of the best individual
            diversity: Population diversity metric
            params: Best parameters (flattened dict)
            symbol: Trading pair
            method: 'genetic' or 'optuna_tpe'
        """
        data = {
            f"{symbol}/fitness": fitness,
            f"{symbol}/diversity": diversity,
            f"{symbol}/generation": gen_id,
            f"{symbol}/method": method,
            "global/gen_id": gen_id,
        }
        
        # Add top params (avoid brain_weights which is huge)
        if params:
            for key, val in params.items():
                if key == "brain_weights":
                    continue
                if isinstance(val, (int, float)):
                    data[f"{symbol}/param_{key}"] = val
        
        if self.is_active:
            try:
                wandb.log(data, step=gen_id)
            except Exception as e:
                logger.debug(f"WandB log error: {e}")
        
        # Always maintain local log
        data["_timestamp"] = time.time()
        self._local_log.append(data)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a single metric convenience wrapper.
        """
        data = {name: value}
        if self.is_active:
            try:
                wandb.log(data, step=step)
            except Exception as e:
                logger.debug(f"WandB metric log error: {e}")
        
        data["_timestamp"] = time.time()
        self._local_log.append(data)
    
    def log_rl_epoch(self, epoch: int, reward: float, 
                     loss: float = 0.0, 
                     symbol: str = "global",
                     extra: Optional[Dict] = None):
        """
        Log Reinforcement Learning training metrics.
        
        Args:
            epoch: Training epoch
            reward: Average reward
            loss: Training loss
            symbol: Symbol or 'global'
            extra: Additional metrics
        """
        data = {
            f"rl/{symbol}/reward": reward,
            f"rl/{symbol}/loss": loss,
            f"rl/{symbol}/epoch": epoch,
        }
        if extra:
            for k, v in extra.items():
                data[f"rl/{symbol}/{k}"] = v
        
        if self.is_active:
            try:
                wandb.log(data, step=epoch)
            except Exception as e:
                logger.debug(f"WandB RL log error: {e}")
        
        data["_timestamp"] = time.time()
        self._local_log.append(data)
    
    def log_efficacy(self, symbol: str, efficacy_ratio: float, 
                     rl_outcome: float, pnl_pct: float):
        """Log manual close efficacy for tracking RL feedback quality."""
        data = {
            f"efficacy/{symbol}/ratio": efficacy_ratio,
            f"efficacy/{symbol}/rl_outcome": rl_outcome,
            f"efficacy/{symbol}/pnl_pct": pnl_pct,
        }
        
        if self.is_active:
            try:
                wandb.log(data)
            except Exception as e:
                logger.debug(f"WandB efficacy log error: {e}")
        
        data["_timestamp"] = time.time()
        self._local_log.append(data)

    def compare_epochs(self, current: Dict, previous: Dict, 
                       symbol: str = "fleet") -> Dict:
        """
        Compare two epochs and log the delta as a WandB table.
        
        Args:
            current: {'fitness': f, 'sharpe': s, 'drawdown': d, ...}
            previous: Same structure
            symbol: Symbol or 'fleet'
            
        Returns:
            Dict with deltas
        """
        deltas = {}
        for key in current:
            if key in previous and isinstance(current[key], (int, float)):
                delta = current[key] - previous[key]
                deltas[f"delta/{symbol}/{key}"] = delta
                deltas[f"current/{symbol}/{key}"] = current[key]
                deltas[f"previous/{symbol}/{key}"] = previous[key]
        
        if self.is_active:
            try:
                # Log as summary table
                wandb.log(deltas)
                
                # Also create a WandB Table for comparison
                columns = ["Metric", "Previous", "Current", "Delta", "Improved?"]
                table_data = []
                for key in current:
                    if key in previous and isinstance(current[key], (int, float)):
                        d = current[key] - previous[key]
                        improved = "✅" if d > 0 else ("⚠️" if d == 0 else "❌")
                        table_data.append([key, previous[key], current[key], d, improved])
                
                table = wandb.Table(columns=columns, data=table_data)
                wandb.log({f"comparison/{symbol}": table})
                
            except Exception as e:
                logger.debug(f"WandB compare error: {e}")
        
        self._local_log.append({
            "_type": "comparison",
            "_symbol": symbol,
            "_timestamp": time.time(),
            **deltas
        })
        
        return deltas
    
    def finish(self):
        """End the current WandB run gracefully."""
        if self.is_active:
            try:
                wandb.finish()
                logger.info(f"📊 WandB Run finished: {self._run_name}")
            except Exception as e:
                logger.debug(f"WandB finish error: {e}")
        
        self._run = None
        self._active = False
    
    def get_local_log(self) -> list:
        """Returns collected local metrics (useful when WandB is offline)."""
        return self._local_log.copy()
    
    def get_local_summary(self) -> Dict:
        """Summarize local logs."""
        if not self._local_log:
            return {"total_entries": 0}
        
        return {
            "total_entries": len(self._local_log),
            "first_timestamp": self._local_log[0].get("_timestamp"),
            "last_timestamp": self._local_log[-1].get("_timestamp"),
        }


# Global Singleton
wandb_tracker = WandBTracker()
