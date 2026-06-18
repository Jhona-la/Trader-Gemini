"""
[OMNI COMPILER] Arquitectura Optimizada para RiskManager
Generado Automáticamente.
"""
from risk.risk_manager import RiskManager

class CompiledRiskManager(RiskManager):
    """
    RiskManager con ADN Lógico Hardcodeado para máxima velocidad.
    Reemplaza if/else dinámicos por constantes evaluadas.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ADN INYECTADO
        self._USE_DYNAMIC_STOPS = False
        
    def _calculate_dynamic_stop_loss(self, symbol, side, current_price, atr):
        if not self._USE_DYNAMIC_STOPS:
            # Optuna decidió que la latencia de procesar stops dinámicos no vale la pena
            # Se usa el default estático
            return super()._calculate_dynamic_stop_loss(symbol, side, current_price, atr)
            
        # Lógica acelerada de stop dinámico
        multiplier = self._get_asset_params(symbol).get("trailing_atr_mult", 1.0)
        if side == "LONG":
            return current_price - (atr * multiplier)
        else:
            return current_price + (atr * multiplier)
