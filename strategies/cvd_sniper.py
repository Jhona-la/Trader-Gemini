import logging
import time
from collections import deque
from typing import List, Union
from core.events import MarketEvent, SignalEvent, SignalType

logger = logging.getLogger("CVDSniper")

class CVDSniperStrategy:
    """
    🟣 [MUTACIÓN 31] CVD Divergence Sniper (Cumulative Volume Delta)
    QUÉ: Detecta divergencias puras entre Precio y CVD en tiempo real, 
         y añade "Flash Liquidity Sniping" vía OBI (Order Book Imbalance).
    POR QUÉ: Si el precio hace un HH pero el CVD baja, hay trampa. Si entra liquidez 
             repentina (OBI extremo), actuamos como depredadores en milisegundos.
    PARA QUÉ: Entradas Microscalping de máxima precisión con latencia <10ms.
    """
    def __init__(self, portfolio, symbol: str = "ALL"):
        self.strategy_id = "CVD_SNIPER"
        self.symbol = symbol
        self.active = True
        self.horizon = "MICROSCALPING"
        self.portfolio = portfolio
        
        # Guardaremos los picos/valles de precio y cvd (rolling_delta)
        self.price_history = {} # symbol -> deque
        self.cvd_history = {}   # symbol -> deque
        self.last_update = {}
        
    def calculate_signals(self, event: MarketEvent) -> Union[List[SignalEvent], SignalEvent, None]:
        if not self.active: return None
        if self.symbol != "ALL" and event.symbol != self.symbol: return None
        
        sym = event.symbol
        metrics = getattr(event, 'microstructure', {})
        if not metrics:
            return None
            
        cvd = metrics['rolling_delta_60s']
        price = event.close_price
        if not price or price <= 0: return None
        
        if sym not in self.price_history:
            self.price_history[sym] = deque(maxlen=300)
            self.cvd_history[sym] = deque(maxlen=300)
            self.last_update[sym] = 0
            
        now = time.time()
        # Muestrear a 1 Hz (1 segundo por tick)
        if now - self.last_update[sym] < 1.0:
            return None
            
        self.price_history[sym].append(price)
        self.cvd_history[sym].append(cvd)
        self.last_update[sym] = now
        
        if len(self.price_history[sym]) < 60:
            return None
            
        # Analizar divergencia (últimos 60s vs previos 60s)
        prices = list(self.price_history[sym])
        cvds = list(self.cvd_history[sym])
        
        # ⚡ FLASH LIQUIDITY SNIPING (OBI Burst) ⚡
        obi = metrics['obi'] # Order Book Imbalance (-1.0 to 1.0)
        cvd_delta_15s = cvds[-1] - cvds[-15] if len(cvds) >= 15 else 0
        
        signals = []
        
        if obi > 0.85 and cvd_delta_15s > 0:
            logger.info(f"⚡ [FLASH SNIPE] OBI Burst LONG en {sym} (OBI: {obi:.2f}).")
            signals.append(SignalEvent(
                strategy_id="FLASH_SNIPER",
                symbol=sym,
                datetime=event.datetime,
                signal_type=SignalType.LONG,
                strength=1.0,
                confidence=0.98,
                horizon=self.horizon,
                metadata={"trigger": "flash_obi_burst", "tp_pct": 0.002, "sl_pct": 0.001}
            ))
            self.price_history[sym].clear()
            self.cvd_history[sym].clear()
            return signals
            
        elif obi < -0.85 and cvd_delta_15s < 0:
            logger.info(f"⚡ [FLASH SNIPE] OBI Burst SHORT en {sym} (OBI: {obi:.2f}).")
            signals.append(SignalEvent(
                strategy_id="FLASH_SNIPER",
                symbol=sym,
                datetime=event.datetime,
                signal_type=SignalType.SHORT,
                strength=1.0,
                confidence=0.98,
                horizon=self.horizon,
                metadata={"trigger": "flash_obi_burst", "tp_pct": 0.002, "sl_pct": 0.001}
            ))
            self.price_history[sym].clear()
            self.cvd_history[sym].clear()
            return signals
        
        p_recent = prices[-15:] # Últimos 15s
        p_past = prices[-60:-15] # Previos 45s
        
        c_recent = cvds[-15:]
        c_past = cvds[-60:-15]
        
        max_p_recent = max(p_recent)
        max_p_past = max(p_past)
        
        max_c_recent = max(c_recent)
        max_c_past = max(c_past)
        
        min_p_recent = min(p_recent)
        min_p_past = min(p_past)
        
        min_c_recent = min(c_recent)
        min_c_past = min(c_past)
        
        signals = []
        
        # 🚨 BEARISH DIVERGENCE (Exhaustion)
        # Price hace Higher High, pero CVD hace Lower High
        if max_p_recent > max_p_past and max_c_recent < max_c_past:
            # Confirmamos con VPIN tóxico o Spoofing
            if metrics['vpin'] > 0.65 or metrics['is_spoofing']:
                logger.info(f"🟣 [CVD DIVERGENCE] BEARISH Exhaustion en {sym}. Price: HH, CVD: LH.")
                signals.append(SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=sym,
                    datetime=event.datetime,
                    signal_type=SignalType.SHORT,
                    strength=0.9,
                    confidence=0.92,
                    horizon=self.horizon,
                    metadata={"trigger": "cvd_bearish_divergence", "tp_pct": 0.003, "sl_pct": 0.0015}
                ))
                # Enfriamiento tras señal
                self.price_history[sym].clear()
                self.cvd_history[sym].clear()
                return signals
                
        # 🟢 BULLISH DIVERGENCE (Absorption)
        # Price hace Lower Low, pero CVD hace Higher Low
        if min_p_recent < min_p_past and min_c_recent > min_c_past:
            # Confirmamos que la gravedad magnética empuja hacia arriba
            if metrics['magnetic_pull_up'] > metrics['magnetic_pull_down'] * 1.5:
                logger.info(f"🟣 [CVD DIVERGENCE] BULLISH Absorption en {sym}. Price: LL, CVD: HL.")
                signals.append(SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=sym,
                    datetime=event.datetime,
                    signal_type=SignalType.LONG,
                    strength=0.9,
                    confidence=0.92,
                    horizon=self.horizon,
                    metadata={"trigger": "cvd_bullish_divergence", "tp_pct": 0.003, "sl_pct": 0.0015}
                ))
                self.price_history[sym].clear()
                self.cvd_history[sym].clear()
                return signals
                
        return None
