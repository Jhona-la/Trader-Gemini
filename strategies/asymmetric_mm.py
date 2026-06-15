import logging
import datetime
from typing import List, Union, Dict, Any
from core.events import MarketEvent, SignalEvent, SignalType

logger = logging.getLogger("AsymmetricMM")

class AsymmetricMMStrategy:
    """
    ⚖️ [MUTACIÓN 27] ASYMMETRIC SPREAD SCALPING (MARKET MAKER)
    Opera en mercados planos de alta entropía. Cuando los bots tendenciales 
    y direccionales se apagan por exceso de ruido, este bot se activa para
    capturar el spread en el micro-rango (Market Making asimétrico).
    """
    
    def __init__(self, portfolio, symbol: str = "ALL"):
        self.strategy_id = "ASYMMETRIC_MM"
        self.symbol = symbol
        self.active = True
        self.horizon = "MICROSCALPING"
        self.portfolio = portfolio
        self.spread_capture_pct = 0.0005 # Capturar 0.05% de spread

    def calculate_signals(self, event: MarketEvent) -> Union[List[SignalEvent], SignalEvent, None]:
        if not self.active:
            return None
            
        if self.symbol != "ALL" and event.symbol != self.symbol:
            return None

        # Solo operar si el Risk Manager ha decretado "Entropía Máxima" (Mercado aburrido/ruidoso)
        global_state = getattr(event, 'global_state', {})
        if not global_state.get('entropy_veto', False):
            # El mercado tiene dirección, no queremos hacer Market Making aquí
            return None
            
        # [ALERTA DE RESTRICCIÓN] One-Way Mode (No podemos tener LONG y SHORT a la vez)
        # Verificamos que NO tengamos posiciones abiertas en este símbolo
        if self.portfolio and self.portfolio.has_position(event.symbol):
            return None
            
        # Emitimos dos señales simultáneas (Bid y Ask separadas del precio actual)
        # El sistema de ejecución deberá encargarse de colocar órdenes LIMITADAS
        # Si una se llena, el gestor de ciclo de vida cancelará la otra (OCO virtual).
        
        logger.info(f"⚖️ [ASYMMETRIC MM] Entropía Alta en {event.symbol}. Desplegando redes Maker.")
        
        signals = []
        
        # Limit Buy (Bid)
        signals.append(SignalEvent(
            strategy_id=self.strategy_id,
            symbol=event.symbol,
            datetime=event.datetime,
            signal_type=SignalType.LONG,
            strength=0.5,
            confidence=0.85, 
            horizon=self.horizon,
            metadata={
                "trigger": "asymmetric_spread",
                "is_maker": True,
                "price_offset_pct": -self.spread_capture_pct,
                "tp_pct": self.spread_capture_pct * 1.5,
                "sl_pct": self.spread_capture_pct * 3.0
            }
        ))
        
        # Limit Sell (Ask)
        signals.append(SignalEvent(
            strategy_id=self.strategy_id,
            symbol=event.symbol,
            datetime=event.datetime,
            signal_type=SignalType.SHORT,
            strength=0.5,
            confidence=0.85, 
            horizon=self.horizon,
            metadata={
                "trigger": "asymmetric_spread",
                "is_maker": True,
                "price_offset_pct": self.spread_capture_pct,
                "tp_pct": self.spread_capture_pct * 1.5,
                "sl_pct": self.spread_capture_pct * 3.0
            }
        ))
        
        return signals
