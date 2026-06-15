from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timezone

from core.events import SignalEvent, SignalType
from strategies.strategy import Strategy

class VolumeSpikeDetector(Strategy):
    """
    ⚡ FASE 9: VOLUME SPIKE DETECTOR (Momentum Hunter)
    Detects abnormal volume spikes that precede violent price movements
    (±0.5-2% in seconds). Ideal for MICROSCALPING.
    """
    def __init__(self, symbol: str):
        super().__init__("VOLUME_SPIKE", symbol)
        self.lookback = 20
        self.spike_threshold = 3.0  # 3x average volume
        self.min_confidence = 0.90
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[SignalEvent]:
        signals = []
        if '1m' not in market_data or len(market_data['1m']) < self.lookback:
            return signals
            
        df = market_data['1m']
        if 'volume' not in df.columns or 'close' not in df.columns:
            return signals
            
        # Get recent volume
        volumes = df['volume'].values[-self.lookback:]
        closes = df['close'].values[-self.lookback:]
        
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[:-1]) if len(volumes) > 1 else 1e-10
        avg_vol = max(avg_vol, 1e-10)
        
        vol_ratio = current_vol / avg_vol
        
        if vol_ratio >= self.spike_threshold:
            # Determine direction based on price action during the spike
            price_delta = closes[-1] - closes[-2]
            
            signal_type = SignalType.LONG if price_delta > 0 else SignalType.SHORT
            
            # Confidence scales with volume ratio, capped at 0.99
            confidence = min(0.99, 0.70 + (vol_ratio / 10.0))
            
            if confidence >= self.min_confidence:
                signals.append(
                    SignalEvent(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        datetime=datetime.now(timezone.utc),
                        signal_type=signal_type,
                        strength=confidence,
                        horizon="MICROSCALPING",
                        metadata={
                            "vol_ratio": vol_ratio,
                            "price_delta": price_delta,
                            "avg_vol": avg_vol,
                            "current_vol": current_vol,
                            # Elite signals need tp_pct and sl_pct
                            "tp_pct": 0.006,  # 0.6% fast TP
                            "sl_pct": 0.003   # 0.3% fast SL (2:1 R:R)
                        }
                    )
                )
                
        return signals
