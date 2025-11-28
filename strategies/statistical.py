import numpy as np
import pandas as pd
import talib # Added for trend calculation
from .strategy import Strategy
from core.events import SignalEvent

class StatisticalStrategy(Strategy):
    """
    Pairs Trading Strategy based on Cointegration / Mean Reversion of the Spread.
    """
    def __init__(self, data_provider, events_queue, pair=('ETH/USDT', 'BTC/USDT'), window=20, z_entry=2.0, z_exit=0.0):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.pair = pair # Tuple of two symbols (Y, X) where Spread = Y - beta*X or Ratio = Y/X
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.invested = 0 # 0 = None, 1 = Long Spread, -1 = Short Spread
        self.last_processed_time = None

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            # Get latest bars for both assets
            y_sym, x_sym = self.pair
            
            # We need enough history to calculate Z-Score
            try:
                bars_y = self.data_provider.get_latest_bars(y_sym, n=self.window)
                bars_x = self.data_provider.get_latest_bars(x_sym, n=self.window)
            except KeyError:
                return # Data not ready yet

            if len(bars_y) < self.window or len(bars_x) < self.window:
                return

            # Deduplication: Check if we already processed this timestamp
            # We use the timestamp of the first symbol in the pair
            current_time = bars_y[-1]['datetime']
            if self.last_processed_time == current_time:
                return
            self.last_processed_time = current_time

            # Extract closes
            y_closes = np.array([b['close'] for b in bars_y])
            x_closes = np.array([b['close'] for b in bars_x])
            
            # Calculate Ratio (Simple version of spread for crypto pairs)
            # In a real model, we would calculate dynamic beta using OLS/Kalman Filter
            ratios = y_closes / x_closes
            
            # Calculate Z-Score
            mean = np.mean(ratios)
            std = np.std(ratios, ddof=1)  # Use sample std (Bessel's correction) for finance
            
            if std == 0:
                return

            current_ratio = ratios[-1]
            z_score = (current_ratio - mean) / std
            
            # Calculate Ratio ADX to detect strong trends in the spread
            # Synthetic High/Low for Ratio
            # Ratio High = High Y / Low X (Max numerator, Min denominator)
            # Ratio Low = Low Y / High X (Min numerator, Max denominator)
            highs_y = np.array([b['high'] for b in bars_y])
            lows_y = np.array([b['low'] for b in bars_y])
            highs_x = np.array([b['high'] for b in bars_x])
            lows_x = np.array([b['low'] for b in bars_x])
            
            ratio_highs = highs_y / lows_x
            ratio_lows = lows_y / highs_x
            ratio_closes = ratios # Already calculated
            
            # Calculate Ratio ADX (may return NaN if insufficient data)
            ratio_adx = talib.ADX(ratio_highs, ratio_lows, ratio_closes, timeperiod=14)[-1]
            
            # Handle NaN: If ADX can't be calculated (not enough data), default to 0 (ranging)
            if np.isnan(ratio_adx):
                ratio_adx = 0  # Assume ranging market if not enough data
            
            # Calculate ATR for volatility-based sizing
            # We use the ATR of the asset we are buying (Y or X)
            closes_y = np.array([b['close'] for b in bars_y])
            atr_y = talib.ATR(highs_y, lows_y, closes_y, timeperiod=14)[-1]
            
            closes_x = np.array([b['close'] for b in bars_x])
            atr_x = talib.ATR(highs_x, lows_x, closes_x, timeperiod=14)[-1]
            timestamp = bars_y[-1]['datetime']

            # PRINT STATS TO TERMINAL (User Request)
            print(f"📊 Stat Strategy {self.pair}: Z-Score={z_score:.2f} Ratio_ADX={ratio_adx:.1f}")

            # Trading Logic
            # Long Spread = Buy Y, Sell X (Expect ratio to go up)
            # Short Spread = Sell Y, Buy X (Expect ratio to go down)
            
            # Filter: Don't mean revert if Trend is too strong (ADX > 30)
            if ratio_adx > 30:
                # Unless Z-Score is EXTREME (e.g. > 4), then maybe it's a climax
                if abs(z_score) < 4.0:
                    return

            if self.invested == 0:
                if z_score < -self.z_entry:
                    # Check Trend for Y (ETH)
                    trend_y = self._get_1h_trend(y_sym)
                    if trend_y == 'DOWN':
                        print(f"  >> Stat Skip {y_sym}: 1h Trend is DOWN")
                    else:
                        # DYNAMIC STRENGTH: Scale based on Z-Score magnitude
                        # Z=2.0 -> 0.5 strength
                        # Z=4.0 -> 0.9 strength
                        z_diff = abs(z_score) - self.z_entry
                        strength = min(1.0, 0.5 + (z_diff * 0.2))
                        
                        
                        print(f"ENTRY LONG SPREAD: Buy {y_sym}, Short {x_sym} (Z={z_score:.2f}, Strength={strength:.2f}, 1h Trend: {trend_y})")
                        self.events_queue.put(SignalEvent(2, y_sym, timestamp, 'LONG', strength=strength, atr=atr_y))
                        self.events_queue.put(SignalEvent(2, x_sym, timestamp, 'SHORT', strength=strength, atr=atr_x))  # ENABLED
                        self.invested = 1
                        
                elif z_score > self.z_entry:
                    # Check Trend for X (BTC)
                    trend_x = self._get_1h_trend(x_sym)
                    if trend_x == 'DOWN':
                        print(f"  >> Stat Skip {x_sym}: 1h Trend is DOWN")
                    else:
                        # DYNAMIC STRENGTH
                        z_diff = abs(z_score) - self.z_entry
                        strength = min(1.0, 0.5 + (z_diff * 0.2))
                        
                        print(f"ENTRY SHORT SPREAD: Short {y_sym}, Buy {x_sym} (Z={z_score:.2f}, Strength={strength:.2f}, 1h Trend: {trend_x})")
                        self.events_queue.put(SignalEvent(2, y_sym, timestamp, 'SHORT', strength=strength, atr=atr_y))  # ENABLED
                        self.events_queue.put(SignalEvent(2, x_sym, timestamp, 'LONG', strength=strength, atr=atr_x))
                        self.invested = -1


            
            elif self.invested == 1:

    def _get_1h_trend(self, symbol):
        """
        Helper to get 1h trend using CLOSED candles.
        """
        try:
            bars_1h = self.data_provider.get_latest_bars_1h(symbol, n=210)
            if len(bars_1h) >= 200:
                closes_1h = np.array([b['close'] for b in bars_1h[:-1]]) # Exclude open candle
                if len(closes_1h) >= 200:
                    ema_50 = talib.EMA(closes_1h, timeperiod=50)[-1]
                    ema_200 = talib.EMA(closes_1h, timeperiod=200)[-1]
                    return 'UP' if ema_50 > ema_200 else 'DOWN'
        except:
            pass
        return 'NEUTRAL'

