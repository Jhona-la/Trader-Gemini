import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ExecutionNettor:
    """
    Virtual Netting Engine - Phase 68
    Intercepta intenciones de orden para evitar 'Pisadas de Patas' (Colisiones).
    Mantiene un Virtual Ledger por horizonte (SCALPING vs SWING) y solo envía
    a Binance la exposición NETA, ahorrando 100% de comisiones en cruces.
    """
    def __init__(self, execution_handler):
        self.execution_handler = execution_handler
        
        # symbol -> { 'SCALPING': +1.0, 'SWING': -0.5 }
        # Positivos = LONG, Negativos = SHORT
        self.virtual_ledger: Dict[str, Dict[str, float]] = {}
        
        # symbol -> net exposure in Binance
        self.actual_binance_exposure: Dict[str, float] = {}
        
    async def execute_order(self, event) -> Dict[str, Any]:
        """
        Intercepta la orden OrderEvent, actualiza el Virtual Ledger y ejecuta el Delta.
        """
        symbol = event.symbol
        side = event.direction  # "BUY" or "SELL" in OrderEvent
        quantity = event.quantity
        order_type = getattr(event, 'order_type', 'MARKET')
        price = getattr(event, 'price', None)
        reduce_only = getattr(event, 'reduce_only', False)
        horizon = getattr(event, 'horizon', 'SCALPING')

        if symbol not in self.virtual_ledger:
            self.virtual_ledger[symbol] = {"SCALPING": 0.0, "SWING": 0.0, "BOTH": 0.0, "MICROSCALPING": 0.0}
            self.actual_binance_exposure[symbol] = 0.0
            
        if horizon not in self.virtual_ledger[symbol]:
            self.virtual_ledger[symbol][horizon] = 0.0

        qty_change = quantity if side.upper() == 'BUY' else -quantity
        
        # Actualizamos Virtual Ledger
        prev_net = sum(self.virtual_ledger[symbol].values())
        self.virtual_ledger[symbol][horizon] += qty_change
        
        # Proteccion contra ceros por precision de flotantes
        if abs(self.virtual_ledger[symbol][horizon]) < 1e-8:
            self.virtual_ledger[symbol][horizon] = 0.0
            
        new_net = sum(self.virtual_ledger[symbol].values())
        if abs(new_net) < 1e-8:
            new_net = 0.0
            
        delta = new_net - self.actual_binance_exposure[symbol]
        
        logger.info(f"🛡️ [NETTING] {symbol} | Req: {horizon} {side} {quantity} | Virtual Net: {prev_net:.4f} -> {new_net:.4f} | Delta a Ejecutar: {delta:.4f}")
        
        # Si el Delta es muy pequeño o cero, la orden se neteó internamente
        if abs(delta) < 1e-8:
            logger.info(f"✅ [NETTING] Orden Compensada Totalmente. Binance ignorado. Ahorro de Fees 100%.")
            return {"status": "NETTED", "virtual_ledger": self.virtual_ledger[symbol]}
            
        # Si hay Delta, enviamos la orden real
        real_side = 'BUY' if delta > 0 else 'SELL'
        real_qty = abs(delta)
        
        # Delegar ejecución física a Binance actualizando el evento
        event.direction = real_side
        event.quantity = real_qty
        
        response = None
        if self.execution_handler:
            import asyncio
            if asyncio.iscoroutinefunction(self.execution_handler.execute_order):
                response = await self.execution_handler.execute_order(event)
            else:
                response = self.execution_handler.execute_order(event)
        
        # Asumiendo ejecución exitosa, alineamos nuestra expectativa de la realidad de Binance
        # En sistemas event-driven asincronos (Mocking), a veces solo devuelve dict, o None si va al stream
        self.actual_binance_exposure[symbol] = new_net
        
        return response
