import logging

logger = logging.getLogger("SCALP-Refinements")

class ScalpRefinements:
    """
    AXIOMA: MÓDULO PREDICCIÓN - MEJORAS ESPECÍFICAS PARA SCALP
    """
    
    @staticmethod
    def vpin_score_multiplier(vpin: float, signal_direction: str, flow_direction: str) -> float:
        """
        MEJORA SCALP-1: VPIN como Multiplicador de Score
        VPIN > 0.7: multiplicar el score de señales en la dirección del flujo por 1.15
        """
        if vpin > 0.7 and signal_direction == flow_direction:
            return 1.15
        elif vpin < 0.3:
            return 0.85
        return 1.0
        
    @staticmethod
    def microseasonality_adjustment(hour_utc: int) -> int:
        """
        MEJORA SCALP-2: MICROSEASONALIDAD POR ACTIVO
        Señales LONG entre 13:00-16:00 UTC reciben +3 puntos (overlap London-NY)
        """
        if 13 <= hour_utc <= 16:
            return 3
        return 0
        
    @staticmethod
    def maker_taker_optimizer(probability_of_fill: float) -> str:
        """
        MEJORA SCALP-3: MAKER-TAKER OPTIMIZER
        Usar LIMIT cuando P(fill < 5s) > 80%, MARKET cuando P(fill < 5s) < 50%
        """
        if probability_of_fill > 0.80:
            return 'LIMIT'
        elif probability_of_fill < 0.50:
            return 'MARKET'
        return 'LIMIT' # Default behavior
        
    @staticmethod
    def tape_speed_analytics(tx_per_second: float, percentile_10: float) -> bool:
        """
        MEJORA SCALP-4: TAPE SPEED ANALYTICS
        Tape seco (< percentil 10 de transacciones/s): no operar
        """
        return tx_per_second >= percentile_10
