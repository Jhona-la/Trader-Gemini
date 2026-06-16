import numpy as np
import time
import logging

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    logging.warning("LightGBM not installed. Surrogate Model will fail. Run: pip install lightgbm")

logger = logging.getLogger("SurrogateEngine")

class ParetoSurrogateEnsemble:
    """
    Motor Subrogado (Surrogate Model) basado en LightGBM.
    Aprende a predecir resultados de backtest en microsegundos, permitiendo 
    evaluar 10 millones de combinaciones virtualmente.
    """
    def __init__(self, n_estimators=200, random_state=42):
        self.is_trained = False
        
        if lgb is None:
            self.models = {}
            return

        # Modelos independientes para optimización Multi-Objetivo (Pareto)
        self.models = {
            'pnl': lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
            'max_dd': lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
            'win_rate': lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        }
        
    def train(self, X_params: np.ndarray, y_pnl: np.ndarray, y_dd: np.ndarray, y_wr: np.ndarray):
        """
        Entrena o actualiza el ensamble de subrogados usando resultados reales de F2/F3/F4.
        """
        if lgb is None: return
        
        t0 = time.perf_counter()
        
        # Training is extremely fast in LightGBM for <10k rows
        self.models['pnl'].fit(X_params, y_pnl)
        self.models['max_dd'].fit(X_params, y_dd)
        self.models['win_rate'].fit(X_params, y_wr)
        
        t1 = time.perf_counter()
        self.is_trained = True
        logger.info(f"🧠 Surrogate Ensemble entrenado sobre {len(X_params)} muestras en {(t1-t0)*1000:.2f} ms")

    def predict(self, X_params: np.ndarray) -> dict:
        """
        Predice los resultados para la matriz de configuraciones pasada.
        Acepta millones de filas.
        """
        if not self.is_trained or lgb is None:
            # Fallback a ruido si no está entrenado (Exploración pura)
            return {
                'pnl': np.random.uniform(-1, 1, len(X_params)),
                'max_dd': np.random.uniform(0, 0.5, len(X_params)),
                'win_rate': np.random.uniform(20, 80, len(X_params))
            }
            
        t0 = time.perf_counter()
        preds = {
            'pnl': self.models['pnl'].predict(X_params),
            'max_dd': self.models['max_dd'].predict(X_params),
            'win_rate': self.models['win_rate'].predict(X_params)
        }
        t1 = time.perf_counter()
        logger.debug(f"⚡ Inferencia de {len(X_params)} configs completada en {(t1-t0)*1000:.2f} ms")
        
        return preds
        
    def filter_promising_configs(self, X_params: np.ndarray, top_k: int = 10000) -> tuple:
        """
        Filtra millones de combinaciones y devuelve el Top K usando la Función de Fitness Subrogada.
        """
        preds = self.predict(X_params)
        
        # Fitness proxy: Maximize PnL, Minimize DD, Maximize WR
        # Avoid division by zero
        safe_dd = np.clip(preds['max_dd'], 0.001, 1.0)
        fitness = preds['pnl'] / safe_dd + (preds['win_rate'] / 100.0)
        
        top_indices = np.argsort(fitness)[::-1][:top_k]
        
        return X_params[top_indices], fitness[top_indices]
