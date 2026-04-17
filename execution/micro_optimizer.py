"""
Optimizador de ejecución para micro cuentas
"""
from typing import Dict

class MicroExecutionOptimizer:
    def __init__(self, micro_awareness):
        self.micro = micro_awareness
        self.fee_optimization = True
        
    def optimize_order(self, symbol: str, side: str, size: float, 
                      price: float, order_type: str) -> Dict:
        """Optimiza órdenes para micro cuenta"""
        # Verificar mínimo notional
        notional = size * price
        min_notional = self.micro.config.MIN_NOTIONAL
        
        if notional < min_notional:
            size = min_notional / price
            
        # Optimizar tipo de orden
        optimal_type = self._get_optimal_order_type(symbol, side, size, price)
        
        return {
            'symbol': symbol,
            'side': side,
            'size': round(size, 6),
            'price': price,
            'type': optimal_type,
            'micro_optimized': True
        }
    
    def _get_optimal_order_type(self, symbol: str, side: str, 
                               size: float, price: float) -> str:
        """Selecciona el mejor tipo de orden para reducir fees"""
        if self.fee_optimization:
            spread = self._get_current_spread(symbol)
            if spread < 0.0008:
                return 'LIMIT'
        return 'MARKET'
        
    def _get_current_spread(self, symbol: str) -> float:
        # Esto debería implementarse para conectar con el orderbook en tiempo real
        # Por ahora retorna un default conservador
        return 0.001
