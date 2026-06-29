import logging
import datetime
from typing import List, Union, Dict, Any
from core.events import MarketEvent, SignalEvent, SignalType

logger = logging.getLogger("VacuumSniper")

class VacuumSniperStrategy:
    """
    🌪️ [MUTACIÓN 26] LIQUIDITY VACUUM ENGINE
    Detecta zonas de vacío de liquidez (Vacuum Zones) en el LOB mediante
    la aceleración direccional del Order Book Imbalance (OBI).
    Entra justo antes de que el spread colapse hacia el vacío.
    """
    
    def __init__(self, symbol: str = "ALL"):
        self.strategy_id = "VACUUM_SNIPER"
        self.symbol = symbol
        self.active = True
        self.horizon = "MICROSCALPING"
        
        # OBI thresholds to detect a vacuum (10:1 ratio equivalent)
        self.vacuum_obi_threshold = 0.85 

    def calculate_signals(self, event: MarketEvent) -> Union[List[SignalEvent], SignalEvent, None]:
        if not self.active:
            return None
            
        if self.symbol != "ALL" and event.symbol != self.symbol:
            return None

        # Obtener métricas de Order Flow inyectadas por BinanceLoader (Mutación 16)
        metrics = getattr(event, 'order_flow_metrics', {})
        if not metrics:
            return None
            
        obi_velocity = metrics['obi_velocity']
        tick_vol = metrics['tick_volatility']
        
        # Un vacío real ocurre cuando el OBI acelera masivamente pero el precio AÚN no ha saltado
        if tick_vol < 0.0005:
            if obi_velocity > self.vacuum_obi_threshold:
                # Compradores han barrido las órdenes Ask limitadas (Vacuum UP)
                logger.critical(f"🌪️⬆️ [VACUUM SNIPER] Vacuum Detectado ALZA en {event.symbol}! OBI: {obi_velocity:.2f}")
                return self._generate_signal(event, SignalType.LONG, obi_velocity)
                
            elif obi_velocity < -self.vacuum_obi_threshold:
                # Vendedores han barrido las órdenes Bid limitadas (Vacuum DOWN)
                logger.critical(f"🌪️⬇️ [VACUUM SNIPER] Vacuum Detectado BAJA en {event.symbol}! OBI: {obi_velocity:.2f}")
                return self._generate_signal(event, SignalType.SHORT, abs(obi_velocity))
                
        return None

    def _generate_signal(self, event: MarketEvent, direction: SignalType, strength: float) -> SignalEvent:
        return SignalEvent(
            strategy_id=self.strategy_id,
            symbol=event.symbol,
            datetime=event.datetime,
            signal_type=direction,
            strength=min(strength, 1.0),
            confidence=0.99, # Micro-Scalping setup de muy alta probabilidad
            horizon=self.horizon,
            metadata={
                "trigger": "vacuum_zone",
                "tp_pct": 0.003, # TP Elástico será manejado por Lifecycle Manager
                "sl_pct": 0.002
            }
        )
