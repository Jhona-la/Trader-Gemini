"""
Estrategias adaptadas para micro cuentas
"""
from typing import Dict
from .technical import TechnicalStrategy
from core.micro_awareness import MicroAccountAwareness

class MicroOptimizedStrategy(TechnicalStrategy):
    def __init__(self, micro_awareness: MicroAccountAwareness):
        super().__init__()
        self.micro = micro_awareness
        
    def generate_micro_signal(self, data: Dict) -> Dict:
        """Genera señal optimizada para micro cuenta"""
        # Señal original
        original_signal = super().generate_signal(data)
        
        if not original_signal:
            return None
            
        # Verificar viabilidad para micro cuenta
        is_viable, reason = self.micro.is_trade_viable(
            self.symbol, data['close'], original_signal['target_profit']
        )
        
        if not is_viable:
            return None
            
        # Calcular tamaño viable
        size, adjusted = self.micro.calculate_viable_trade_size(
            self.symbol, data['close']
        )
        
        return {
            **original_signal,
            'size': size,
            'micro_optimized': True,
            'size_adjusted': adjusted,
            'min_target': self.micro.calculate_breakeven_threshold(size, data['close']) * 1.5
        }
