"""
[OMNI COMPILER] Arquitectura Optimizada para StatisticalStrategy (Patterns)
Generado Automáticamente.
"""
from strategies.statistical import StatisticalStrategy

class CompiledPatternStrategy(StatisticalStrategy):
    """
    StatisticalStrategy con ADN Lógico inyectado.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._STRICT_WICK_FILTER = False
        
    def validate_wick_structure(self, candle):
        if not self._STRICT_WICK_FILTER:
            # Optuna descubrió que las mechas estrictas reducen el WinRate global
            return True
        return super().validate_wick_structure(candle)
