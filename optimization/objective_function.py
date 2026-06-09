import numpy as np
import math
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger("ObjectiveFunction")

class WalkForwardValidator:
    """
    Divide los datos históricos en K ventanas cronológicas (folds) OOS (Out-of-Sample).
    """
    def __init__(self, k_folds: int = 5, purge_bars: int = 50):
        self.k_folds = k_folds
        self.purge_bars = purge_bars

    def split_data(self, total_bars: int) -> List[Tuple[int, int]]:
        if total_bars < self.k_folds * 100:
            logger.warning("No hay suficientes barras para un Walk-Forward riguroso.")
            return [(0, total_bars)]
            
        fold_size = total_bars // self.k_folds
        folds = []
        for i in range(self.k_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            folds.append((start_idx, end_idx))
        return folds

class ObjectiveFunction:
    """
    El Corazón Matemático de la Búsqueda Suprema.
    """
    def __init__(self, ruin_threshold_pct: float = -0.40, max_p_ruin: float = 0.05, regime_penalty_alpha: float = 0.5):
        self.ruin_threshold_pct = ruin_threshold_pct 
        self.max_p_ruin = max_p_ruin
        self.alpha = regime_penalty_alpha

    def calculate_cycle_returns(self, trades: List[Any]) -> float:
        """Suma de retornos de los trades en un ciclo/fold"""
        return sum([t.pnl_pct if hasattr(t, 'pnl_pct') else t.get('pnl_pct', 0.0) for t in trades])

    def survival_filter(self, cycle_returns: List[float]) -> bool:
        """
        COMPONENTE 1 - Filtro de Supervivencia.
        """
        if not cycle_returns:
            return False
            
        n = len(cycle_returns)
        ruin_events = sum(1 for r in cycle_returns if r <= self.ruin_threshold_pct)
        
        if n < 10:
            # Para muestras pequeñas (ej. 5 folds), Agresti-Coull siempre dará un límite superior alto.
            # Usar frecuencia empírica directa o permitir al menos 0 o 1 fallos dependiendo de la rigurosidad.
            empirical_p = ruin_events / n
            if empirical_p > self.max_p_ruin:
                logger.debug(f"💀 Fallo de Supervivencia Empírico: {empirical_p*100:.1f}% > Límite {self.max_p_ruin*100:.1f}%")
                return False
            return True
            
        # Agresti-Coull interval (más seguro para muestras grandes)
        z = 1.96 # 95% conf
        n_tilde = n + z**2
        p_tilde = (ruin_events + 0.5 * z**2) / n_tilde
        upper_bound = p_tilde + z * math.sqrt(max(0, p_tilde * (1 - p_tilde) / n_tilde))
        
        if upper_bound > self.max_p_ruin:
            logger.debug(f"💀 Fallo de Supervivencia Estricto: P(Ruina) Upper Bound {upper_bound*100:.1f}% > Límite {self.max_p_ruin*100:.1f}%")
            return False
        return True

    def geometric_mean(self, returns: List[float]) -> float:
        """
        COMPONENTE 2 - Media Geométrica.
        """
        if not returns:
            return 0.0
            
        product = 1.0
        for r in returns:
            val = max(1.0 + r, 0.001)
            product *= val
            
        n = len(returns)
        g_mean = (product ** (1.0 / n)) - 1.0
        return float(g_mean)

    def regime_inconsistency_penalty(self, g_mean_global: float, g_mean_by_regime: Dict[str, float]) -> float:
        """
        COMPONENTE 3 - Castigo por Inconsistencia.
        """
        if not g_mean_by_regime or len(g_mean_by_regime) <= 1:
            return g_mean_global
            
        g_values = list(g_mean_by_regime.values())
        mean_g_regimes = float(np.mean(g_values))
        if mean_g_regimes == 0:
            return g_mean_global
            
        std_g_regimes = float(np.std(g_values))
        cov_regimes = std_g_regimes / abs(mean_g_regimes)
        
        adjusted_score = g_mean_global * (1.0 - self.alpha * cov_regimes)
        return float(adjusted_score)

    def evaluate_configuration(self, fold_results: List[Dict]) -> Tuple[float, Dict[str, Any]]:
        """
        Evalúa una configuración candidata.
        Retorna (Score, Metrics)
        """
        if not fold_results:
            return -999.0, {"error": "No results"}
            
        cycle_returns = [self.calculate_cycle_returns(f.get('trades', [])) for f in fold_results]
        
        if not self.survival_filter(cycle_returns):
            return -999.0, {"error": "Survival Filter Failed"}
            
        g_mean = self.geometric_mean(cycle_returns)
        
        regime_returns = {}
        for f, ret in zip(fold_results, cycle_returns):
            reg = f.get('regime', 'UNKNOWN')
            if reg not in regime_returns:
                regime_returns[reg] = []
            regime_returns[reg].append(ret)
            
        g_mean_by_regime = {reg: self.geometric_mean(rets) for reg, rets in regime_returns.items()}
        
        final_score = self.regime_inconsistency_penalty(g_mean, g_mean_by_regime)
        
        cumulative = 1.0
        cycles_to_100 = len(cycle_returns) + 1
        for i, ret in enumerate(cycle_returns):
            cumulative *= (1.0 + ret)
            if cumulative >= 2.0:
                cycles_to_100 = i + 1
                break
                
        metrics = {
            "g_mean": g_mean,
            "final_score": final_score,
            "cycles_to_100": cycles_to_100,
            "g_by_regime": g_mean_by_regime,
            "cycle_returns": cycle_returns
        }
        
        return final_score, metrics
