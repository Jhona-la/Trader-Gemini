import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class OptimizationLevel:
    LEVEL_A = "NIVEL_A_CRITICOS"
    LEVEL_B = "NIVEL_B_ESTRATEGIA"
    LEVEL_C = "NIVEL_C_FEATURES"
    LEVEL_D = "NIVEL_D_ORQUESTACION"

class SearchSpace:
    """
    Define el Espacio de Búsqueda Jerárquico y aplica la lógica de No-Colisión (Capa 2).
    """
    
    def __init__(self):
        # Rangos autorizados dictados por Nivel A (Valores Fijos o Límites Absolutos)
        self.bounds = {
            OptimizationLevel.LEVEL_A: {
                "tp_pct": (0.005, 0.05),
                "sl_pct": (0.002, 0.02),
                "leverage_multiplier": (1, 10),
            },
            OptimizationLevel.LEVEL_B: {
                "trend_confirmation_threshold": (0.0, 5.0),
                "macd_slow": (21, 50),
            },
            OptimizationLevel.LEVEL_C: {
                "rsi_window": (7, 21),
                "macd_fast": (8, 20),
                "rsi_oversold": (20, 35),
                "rsi_overbought": (65, 80)
            },
            OptimizationLevel.LEVEL_D: {
                "ml_lookback_bars": (100, 500)
            }
        }
        
    def get_space(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        return self.bounds

    def no_colision(self, config: Dict[str, Any]) -> bool:
        for level, params in self.bounds.items():
            for p_name, (p_min, p_max) in params.items():
                if p_name in config:
                    val = config[p_name]
                    if val < p_min or val > p_max:
                        logger.warning(f"❌ [CAPA 2] Colisión: {p_name}={val} fuera de rango [{p_min}, {p_max}]")
                        return False

        if "macd_fast" in config and "macd_slow" in config:
            if config["macd_fast"] >= config["macd_slow"]:
                return False
        if "sl_pct" in config and "tp_pct" in config:
            if config["sl_pct"] * 0.5 > config["tp_pct"]:
                return False
                
        return True
