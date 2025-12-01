import talib
import numpy as np
from .strategy import Strategy
from core.events import SignalEvent

class PatternStrategy(Strategy):
    """
    Pattern Recognition Strategy.
    Uses TA-Lib to detect high-probability candlestick patterns.
    
    Patterns:
    1. Bullish Engulfing (Reversal)
    2. Hammer (Reversal)
    3. Morning Star (Reversal)
    
    Filters:
    - Only trades in direction of 1h Trend.
    - Requires RSI < 50 (Pullback context) for reversals.
    """
    def __init__(self, data_provider, events_queue):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.symbol_list = data_provider.symbol_list
        # Cooldown to avoid spamming signals for the same pattern
        self.last_signal_time = {s: None for s in self.symbol_list}

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.data_provider.get_latest_bars(s, n=50)
                if len(bars) < 50:
                    continue
                
                # Extract OHLC
                opens = np.array([b['open'] for b in bars])
                highs = np.array([b['high'] for b in bars])
                lows = np.array([b['low'] for b in bars])
                closes = np.array([b['close'] for b in bars])
                
                # 1. Detect Patterns (TA-Lib returns 100 for Bullish, -100 for Bearish, 0 for None)
                engulfing = talib.CDLENGULFING(opens, highs, lows, closes)
                hammer = talib.CDLHAMMER(opens, highs, lows, closes)
                shooting_star = talib.CDLSHOOTINGSTAR(opens, highs, lows, closes)
                morning_star = talib.CDLMORNINGSTAR(opens, highs, lows, closes)
                evening_star = talib.CDLEVENINGSTAR(opens, highs, lows, closes)
                
                # Check LAST CLOSED candle for pattern ([-2])
                # We do NOT use [-1] because it is the current OPEN candle and the pattern might disappear before close.
                # Using [-2] ensures the pattern is confirmed.
                is_bullish_engulfing = engulfing[-2] == 100
                is_bearish_engulfing = engulfing[-2] == -100
                is_hammer = hammer[-2] == 100
                is_shooting_star = shooting_star[-2] == 100
                is_morning_star = morning_star[-2] == 100
                is_evening_star = evening_star[-2] == 100
                
                if not (is_bullish_engulfing or is_bearish_engulfing or is_hammer or is_shooting_star or is_morning_star or is_evening_star):
                    continue
                    
                # 2. Context Filters
                # Get 1h Trend
                trend_1h = 'NEUTRAL'
                try:
                    bars_1h = self.data_provider.get_latest_bars_1h(s, n=210)
                    if len(bars_1h) >= 200:
                        closes_1h = np.array([b['close'] for b in bars_1h[:-1]])
                        ema_50 = talib.EMA(closes_1h, timeperiod=50)[-1]
                        ema_200 = talib.EMA(closes_1h, timeperiod=200)[-1]
                        trend_1h = 'UP' if ema_50 > ema_200 else 'DOWN'
                except:
                    pass
                
                # RSI Filter (Don't buy at top)
                rsi = talib.RSI(closes, timeperiod=14)[-1]
                
                # 3. Signal Logic
                signal_type = None
                pattern_name = ""
                strength = 0.0
                
                if trend_1h == 'UP':
                    if is_bullish_engulfing and rsi < 60:
                        signal_type = 'LONG'
                        pattern_name = "Bullish Engulfing"
                        strength = 0.8
                    elif is_hammer and rsi < 45:
                        signal_type = 'LONG'
                        pattern_name = "Hammer"
                        strength = 0.7
                    elif is_morning_star and rsi < 45:
                        signal_type = 'LONG'
                        pattern_name = "Morning Star"
                        strength = 0.9
                
                elif trend_1h == 'DOWN':
                    if is_bearish_engulfing and rsi > 40:
                        signal_type = 'SHORT'
                        pattern_name = "Bearish Engulfing"
                        strength = 0.8
                    elif is_shooting_star and rsi > 55:
                        signal_type = 'SHORT'
                        pattern_name = "Shooting Star"
                        strength = 0.7
                    elif is_evening_star and rsi > 55:
                        signal_type = 'SHORT'
                        pattern_name = "Evening Star"
                        strength = 0.9
                
                # 4. Emit Signal
                if signal_type:
                    timestamp = bars[-1]['datetime']
                    
                    # Dedup: Don't signal same candle twice
                    if self.last_signal_time[s] == timestamp:
                        continue
                        
                    self.last_signal_time[s] = timestamp
                    
                    # Calculate ATR for risk management
                    atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                    
                    print(f"[PATTERN] Strategy {s}: {pattern_name} detected! (RSI={rsi:.1f}, 1h={trend_1h})")
                    self.events_queue.put(SignalEvent(4, s, timestamp, signal_type, strength=strength, atr=atr))
