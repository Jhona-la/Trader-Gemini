import logging
import time
from typing import List, Union
from core.events import MarketEvent, SignalEvent, SignalType

logger = logging.getLogger("DarkPoolSurfer")

class DarkPoolSurferStrategy:
    """
    🐋 [MUTACIÓN 34] Dark Pool Surfer (Piggyback Strategy)
    QUÉ: Estrategia ultrarrápida que detecta firmas TWAP / Icebergs institucionales
         agrupados temporalmente a nivel milisegundos.
    POR QUÉ: Los HFT y ballenas usan algoritmos ejecutando órdenes idénticas 
             velozmente sin mover el LOB. El rastro ("prints") anticipa un gran sweep.
    PARA QUÉ: Entrar en micro-scalping colgado de la ballena y salir en milisegundos.
    """
    def __init__(self, portfolio, symbol: str = "ALL"):
        self.strategy_id = "DARK_POOL_SURFER"
        self.symbol = symbol
        self.active = True
        self.horizon = "MICROSCALPING"
        self.portfolio = portfolio
        self.last_signal_time = {}
        self.cooldown = 3.0 # Segundos entre surfeos
        
    def calculate_signals(self, event: MarketEvent) -> Union[List[SignalEvent], SignalEvent, None]:
        if not self.active: return None
        if self.symbol != "ALL" and event.symbol != self.symbol: return None
        
        sym = event.symbol
        metrics = getattr(event, 'microstructure', {})
        if not metrics:
            return None
            
        is_dark_pool = metrics['is_dark_pool_print']
        dark_side = metrics['dark_pool_side']
        
        if not is_dark_pool or not dark_side:
            return None
            
        now = time.time()
        if now - self.last_signal_time.get(sym, 0) < self.cooldown:
            return None
            
        self.last_signal_time[sym] = now
        
        signal_type = SignalType.LONG if dark_side == "BUY" else SignalType.SHORT
        
        logger.info(f"🐋 [DARK POOL] Firma Institucional detectada en {sym}. Lado: {dark_side}. ¡Cabalagando ola!")
        
        return SignalEvent(
            strategy_id=self.strategy_id,
            symbol=sym,
            datetime=event.datetime,
            signal_type=signal_type,
            strength=1.0,
            confidence=0.95,
            horizon=self.horizon,
            metadata={
                "trigger": "dark_pool_twap_detected", 
                "tp_pct": 0.0015, # TP muy pequeño, salir rápido
                "sl_pct": 0.0010
            }
        )
