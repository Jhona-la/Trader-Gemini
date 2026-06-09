from core.evolutionary_calibrator import evolutionary_calibrator
from strategies.technical import HybridScalpingStrategy
from config import Config
from utils.logger import logger

class SwingMotor(HybridScalpingStrategy):
    """
    Motor Especializado de Estructura Macro (Fase 2).
    Hereda de la estrategia híbrida técnica pero aplica adaptabilidad
    orientada a la tendencia y a la paciencia estructural.
    """
    def __init__(self, data_provider, events_queue, genotype=None, priority=2):
        # Forzamos el horizonte a SWING
        super().__init__(data_provider, events_queue, genotype=genotype, horizon="SWING", priority=priority)
        self.strategy_id = f"[SWG_MOTOR]_{self.symbol}" if self.symbol else f"[SWG_MOTOR]"
        logger.info(f"🌊 [SWING_MOTOR] Inicializado para {self.symbol}. Buscando rupturas estructurales.")

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
        hurdle_multiplier = evolutionary_calibrator.get_hurdle_multiplier(symbol, "SWING")
        
        # 2. Modificar temporalmente el STRENGTH_THRESHOLD
        original_strength = self.STRENGTH_THRESHOLD
        
        # En Swing, la fricción controla cuánta fuerza direccional necesitamos
        new_strength = original_strength * hurdle_multiplier
        self.STRENGTH_THRESHOLD = min(0.95, max(0.20, new_strength))
        
        if hurdle_multiplier != 1.0:
            logger.debug(f"🧬 [SWG_EVOL] {symbol} Threshold ajustado: {original_strength:.2f} -> {self.STRENGTH_THRESHOLD:.2f} (x{hurdle_multiplier:.2f})")

        try:
            # 3. Delegar la evaluación a la clase base
            super().generate_signals(event)
        finally:
            # Restaurar
            self.STRENGTH_THRESHOLD = original_strength
            
    def update_recursive_weights(self, trade_outcome):
        """
        Registra el resultado en el calibrador evolutivo.
        """
        super().update_recursive_weights(trade_outcome)
        
        if isinstance(trade_outcome, float):
            pnl = trade_outcome
            symbol = self.symbol
        else:
            pnl = trade_outcome.pnl if hasattr(trade_outcome, 'pnl') else (trade_outcome.exit_price - trade_outcome.entry_price) * trade_outcome.direction
            symbol = getattr(trade_outcome, 'symbol', self.symbol)
            
        if symbol:
            evolutionary_calibrator.register_trade_outcome(symbol, "SWING", pnl)
