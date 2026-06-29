from core.evolutionary_calibrator import evolutionary_calibrator
from strategies.technical import HybridScalpingStrategy
from config import Config
from utils.logger import logger

class ScalpingMotor(HybridScalpingStrategy):
    """
    Motor Especializado de Alta Frecuencia (Fase 2).
    Hereda de la estrategia híbrida técnica pero aplica restricciones extremas
    y adaptabilidad evolutiva diseñada para alcanzar el 100% de Win Rate.
    """
    def __init__(self, data_provider, events_queue, genotype=None, priority=1):
        # Forzamos el horizonte a SCALPING
        super().__init__(data_provider, events_queue, genotype=genotype, horizon="SCALPING", priority=priority)
        self.strategy_id = f"[SCL_MOTOR]_{self.symbol}" if self.symbol else f"[SCL_MOTOR]"
        logger.info(f"🚀 [SCALPING_MOTOR] Inicializado para {self.symbol}. Buscando precisión sub-nanosegundo.")

    def generate_signals(self, event):
        """
        Intercepta la generación de señales para aplicar el multiplicador evolutivo.
        """
        symbol = self.symbol
        if not symbol and event and getattr(event, 'symbol', None):
            symbol = event.symbol

        if not symbol:
            # Fallback to base class if no symbol can be determined
            super().generate_signals(event)
            return

        # 1. Obtener multiplicador evolutivo basado en Win Rate reciente
        hurdle_multiplier = evolutionary_calibrator.get_hurdle_multiplier(symbol, "SCALPING")
        
        # 2. Modificar temporalmente el STRENGTH_THRESHOLD
        original_strength = self.STRENGTH_THRESHOLD
        
        # Aplicamos la fricción evolutiva (más alto = más restrictivo = menos trades pero más seguros)
        new_strength = original_strength * hurdle_multiplier
        # No permitir que exceda 0.95 ni caiga por debajo de 0.30
        self.STRENGTH_THRESHOLD = min(0.95, max(0.30, new_strength))
        
        if hurdle_multiplier != 1.0:
            logger.debug(f"🧬 [SCL_EVOL] {symbol} Threshold ajustado: {original_strength:.2f} -> {self.STRENGTH_THRESHOLD:.2f} (x{hurdle_multiplier:.2f})")

        try:
            # 3. Delegar la evaluación a la clase base
            super().generate_signals(event)
        finally:
            # Restaurar el valor original para no contaminar otras llamadas
            self.STRENGTH_THRESHOLD = original_strength
            
    def update_recursive_weights(self, trade_outcome):
        """
        Registra el resultado en el calibrador evolutivo y luego delega
        al motor de memoria subconsciente (cognitive_memory).
        """
        super().update_recursive_weights(trade_outcome)
        
        if isinstance(trade_outcome, float):
            pnl = trade_outcome
            symbol = self.symbol
        else:
            pnl = trade_outcome.pnl if hasattr(trade_outcome, 'pnl') else (trade_outcome.exit_price - trade_outcome.entry_price) * trade_outcome.direction
            symbol = getattr(trade_outcome, 'symbol', self.symbol)
            
        if symbol:
            evolutionary_calibrator.register_trade_outcome(symbol, "SCALPING", pnl)
