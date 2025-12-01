import talib
import numpy as np
from .strategy import Strategy
from core.events import SignalEvent

class TechnicalStrategy(Strategy):
    def __init__(self, data_provider, events_queue):
        self.data_provider = data_provider
        self.events_queue = events_queue
        
        # RELAXED THRESHOLDS: Allow more opportunities while maintaining trend filter
        # RSI 45 allows entry on pullbacks in uptrends (was 30, too restrictive)
        # RSI 35/70 - More selective to avoid chop
        self.rsi_period = 14
        self.rsi_buy = 35  # Was 45 - Stricter entry
        self.rsi_sell = 70 # Was 65 - Let winners run slightly more
        self.symbol_list = data_provider.symbol_list
        self.bought = {s: False for s in self.symbol_list}
        self.last_processed_times = {s: None for s in self.symbol_list}

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            for s in self.symbol_list:
                # Get enough bars for EMA 200
                bars = self.data_provider.get_latest_bars(s, n=210)
                if len(bars) < 210:
                    continue  # Skip this symbol, continue with next one

                # Extract close prices
                closes = np.array([b['close'] for b in bars])
                
                # CRITICAL FIX: Use CLOSED candle ([-2]) for indicators to avoid Repainting/Lookahead Bias
                # If we use [-1], the signal might flash ON and OFF during the candle.
                # We want confirmed signals.
                
                # Calculate RSI
                rsi = talib.RSI(closes, timeperiod=self.rsi_period)
                current_rsi = rsi[-2] # Use last CLOSED candle
                
                # TREND FILTER: Calculate EMA 50 and EMA 200
                ema_50 = talib.EMA(closes, timeperiod=50)
                ema_200 = talib.EMA(closes, timeperiod=200)
                
                # Trend determination: Uptrend if EMA 50 > EMA 200 (on closed candle)
                in_uptrend = ema_50[-2] > ema_200[-2]
                
                timestamp = bars[-1]['datetime']

                # Deduplication
                if self.last_processed_times[s] == timestamp:
                    continue
                self.last_processed_times[s] = timestamp
                
                # STRATEGY COLLABORATION: Regime Detection
                # Calculate ADX to detect trend strength
                highs = np.array([b['high'] for b in bars])
                lows = np.array([b['low'] for b in bars])
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
                current_adx = adx[-2] if len(adx) > 1 else 20 # Last CLOSED candle
                
                # Calculate ATR for volatility-based sizing
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                current_atr = atr[-2] if len(atr) > 1 else 0 # Last CLOSED candle
                
                # VOLUME CONFIRMATION: Check if volume supports signal
                volumes = np.array([b['volume'] for b in bars])
                volume_ma = talib.SMA(volumes, timeperiod=20)[-2]
                current_volume = bars[-2]['volume'] # Last CLOSED volume
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
                
                # MOMENTUM BREAKOUT CHECK (Before ADX Filter)
                # If ADX > 25 (Strong Trend) AND RSI is healthy (50-70) AND Volume Surge
                # We want to capture this move, not skip it!
                is_breakout = (current_adx > 25 and 
                               50 < current_rsi < 70 and 
                               volume_ratio > 1.5)
                
                # DYNAMIC RSI THRESHOLDS (Regime Adaptive)
                # User Request: "Keep moving with the trend"
                
                current_rsi_buy = self.rsi_buy
                current_rsi_sell = self.rsi_sell
                
                if current_adx > 25:
                    # TRENDING REGIME: Aggressive Entries on Pullbacks
                    # In strong trends, RSI rarely drops to 35. We enter on shallow pullbacks.
                    if in_uptrend:
                        current_rsi_buy = 50  # Buy shallow dips (was 35)
                        current_rsi_sell = 75 # Let winners run (was 70)
                        # print(f"  🚀 {s}: Trending Bull (ADX {current_adx:.1f}) - Aggressive Entry < {current_rsi_buy}")
                    else:
                        # Downtrend logic (if we were shorting, but we focus on Longs for now)
                        pass
                else:
                    # RANGING REGIME: Conservative Mean Reversion
                    # Market is choppy, require deep value
                    current_rsi_buy = 35
                    current_rsi_sell = 70
                    # print(f"  🦀 {s}: Ranging (ADX {current_adx:.1f}) - Conservative Entry < {current_rsi_buy}")

                # Skip strong trending markets ONLY if we are counter-trend trading (which we aren't)
                # We removed the 'continue' block here to allow Trend Following.

                # MULTI-TIMEFRAME ANALYSIS: 5m, 15m, 1h
                # Calculate RSI on multiple timeframes for confluence
                rsi_5m = 50  # Default neutral
                rsi_15m = 50
                rsi_1h = 50
                
                try:
                    # 5m timeframe
                    bars_5m = self.data_provider.get_latest_bars_5m(s, n=30)
                    if len(bars_5m) >= 14:
                        closes_5m = np.array([b['close'] for b in bars_5m])
                        rsi_5m = talib.RSI(closes_5m, timeperiod=14)[-1]
                except:
                    pass
                
                try:
                    # 15m timeframe
                    bars_15m = self.data_provider.get_latest_bars_15m(s, n=30)
                    if len(bars_15m) >= 14:
                        closes_15m = np.array([b['close'] for b in bars_15m])
                        rsi_15m = talib.RSI(closes_15m, timeperiod=14)[-1]
                except:
                    pass
                # Result: -1.0 (all overbought) to +1.0 (all oversold)
                
                # 1h Trend (Restored)
                trend_1h = 'NEUTRAL'
                try:
                    bars_1h = self.data_provider.get_latest_bars_1h(s, n=210)
                    if len(bars_1h) >= 200:
                        # Exclude last bar (current open candle)
                        closes_1h = np.array([b['close'] for b in bars_1h[:-1]])
                        
                        if len(closes_1h) >= 200:
                            ema_50_1h = talib.EMA(closes_1h, timeperiod=50)[-1]
                            ema_200_1h = talib.EMA(closes_1h, timeperiod=200)[-1]
                            
                            if ema_50_1h > ema_200_1h:
                                trend_1h = 'UP'
                            else:
                                trend_1h = 'DOWN'
                            
                current_price = bars[-1]['close']
                atr_pct = (current_atr / current_price) * 100 if current_price > 0 else 0.5
                
                # High volatility -> More aggressive (higher RSI threshold)
                # Low volatility -> Conservative (lower RSI threshold)
                if atr_pct > 1.0:  # High volatility
                    dynamic_rsi_buy = 50  # Buy on moderate pullback
                    dynamic_rsi_sell = 60  # Take profit earlier
                elif atr_pct > 0.5:  # Medium volatility
                    dynamic_rsi_buy = self.rsi_buy  # Use default (45)
                    dynamic_rsi_sell = self.rsi_sell  # Use default (65)
                else:  # Low volatility
                    dynamic_rsi_buy = 40  # Wait for deeper pullback
                    dynamic_rsi_sell = 70  # Let it run longer
                
                # VOLUME CONFIRMATION (Calculated above)

                # IMPROVED: Proper position management with trend filter
                # BUY when RSI pullback AND in uptrend (ranging/weak trend markets only)
                # AND 1h Trend is UP or NEUTRAL (Multi-timeframe confirmation)
                
                # 1. RSI PULLBACK ENTRY (Standard)
                is_pullback = current_rsi < dynamic_rsi_buy and in_uptrend
                
                # 2. MOMENTUM BREAKOUT ENTRY (Already calculated above)
                # is_breakout = ...
                
                # ENTRY LONG
                if (is_pullback or is_breakout) and not self.bought[s]:
                    # AGGRESSIVE MODE: Much lower confluence threshold (user requested)
                    # Changed from 0.5 to -0.3 to allow trading in any direction
                    if confluence < -0.3:
                        print(f"  >> Skipping {s}: Weak confluence ({confluence:+.2f}) - extreme disagreement")
                    # REMOVED 1h trend filter for more aggressive trading
                    elif False:  # Disabled
                        pass
                    else:
                        # DYNAMIC STRENGTH: Scale based on signal type
                        if is_breakout:
                            strength = 1.0 # Breakouts are high conviction
                            print(f"[!!] BREAKOUT Signal for {s}: RSI={current_rsi:.1f} Vol={volume_ratio:.1f}x")
                        else:
                            # Pullback strength based on RSI depth
                            rsi_diff = max(0, dynamic_rsi_buy - current_rsi)
                            strength = min(1.0, 0.5 + (rsi_diff * 0.02))
                        
                        # VOLUME BOOST: Increase strength if high volume
                        if volume_ratio > 1.5:
                            strength = min(1.0, strength * 1.2)
                            print(f"  📈 Volume Boost: {volume_ratio:.2f}x avg (strength +20%)")
                        elif volume_ratio < 0.7:
                            # Reduce strength if low volume (weak signal)
                            strength *= 0.8
                            print(f"  📉 Low Volume: {volume_ratio:.2f}x avg (strength -20%)")
                        
                        # CONFLUENCE BOOST: Higher confluence = Higher strength
                        if confluence >= 0.9:  # Near perfect agreement
                            strength = min(1.0, strength * 1.3)  # Perfect confluence +30%
                            print(f"  ✨ Perfect Confluence {confluence:+.2f}: All timeframes agree (strength +30%)")
                        elif confluence >= 0.7:  # Strong agreement
                            strength = min(1.0, strength * 1.15)  # Strong confluence +15%
                            print(f"  🌟 Strong Confluence {confluence:+.2f} (strength +15%)")
                        
                        print(f"✅ BUY SIGNAL! {s} RSI:{current_rsi:.2f} Adaptive_Threshold:{dynamic_rsi_buy} (Strength={strength:.2f}, ATR%={atr_pct:.2f}, Vol={volume_ratio:.2f}x)")
                        signal = SignalEvent(1, s, timestamp, 'LONG', strength=strength, atr=current_atr)
                        self.events_queue.put(signal)
                        self.bought[s] = True
                
                # ENTRY SHORT (Downtrend + Overbought)
                elif not in_uptrend and current_rsi > dynamic_rsi_sell and not self.bought[s]:
                    # Only SHORT if 1h trend is also DOWN or NEUTRAL
                    if trend_1h == 'UP':
                        print(f"  ⏭️  Skipping SHORT {s}: 1h Trend is UP (Counter-trend)")
                    else:
                        # Dynamic strength for SHORT
                        rsi_diff = max(0, current_rsi - dynamic_rsi_sell)
                        strength = min(1.0, 0.5 + (rsi_diff * 0.02))
                        
                        # Volume boost
                        if volume_ratio > 1.5:
                            strength = min(1.0, strength * 1.2)
                            
                        print(f"✅ SELL SIGNAL! {s} RSI:{current_rsi:.2f} Adaptive_Threshold:{dynamic_rsi_sell} (Strength={strength:.2f})")
                        signal = SignalEvent(1, s, timestamp, 'SHORT', strength=strength, atr=current_atr)
                        self.events_queue.put(signal)
                        self.bought[s] = True # Mark as invested (short)
                        
                # EXIT SHORT (oversold or trend reverses)
                elif self.bought[s] and not in_uptrend:
                    if current_rsi < dynamic_rsi_buy:
                        print(f"COVER SHORT SIGNAL! {s} RSI: {current_rsi} (Oversold, Threshold={dynamic_rsi_buy})")
                        signal = SignalEvent(1, s, timestamp, 'EXIT', strength=1.0)
                        self.events_queue.put(signal)
                        self.bought[s] = False
                    elif in_uptrend:
                        print(f"COVER SHORT SIGNAL! {s} Trend reversed (EMA 50 > EMA 200)")
                        signal = SignalEvent(1, s, timestamp, 'EXIT', strength=1.0)
                        self.events_queue.put(signal)
                        self.bought[s] = False

