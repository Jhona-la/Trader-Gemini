import numpy as np
import pandas as pd
import talib # Added for trend calculation
from utils.math_helpers import safe_div
from .strategy import Strategy
from core.events import SignalEvent
from core.enums import EventType, SignalType
from config import Config
from utils.cooldown_manager import cooldown_manager
from utils.analytics import AnalyticsEngine
from core.data_handler import get_data_handler
from utils.logger import logger
from utils.statistics_pro import StatisticsPro

class StatisticalStrategy(Strategy):
    """
    Pairs Trading Strategy based on Cointegration / Mean Reversion of the Spread.
    """
    def __init__(self, data_provider, events_queue, portfolio=None, pair=('ETH/USDT', 'BTC/USDT')):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.portfolio = portfolio # Needed for state tracking
        self.pair = pair # Tuple of two symbols (Y, X) where Spread = Y - beta*X or Ratio = Y/X
        self.window = Config.Strategies.STAT_WINDOW
        self.z_entry = Config.Strategies.STAT_Z_ENTRY
        self.z_exit = Config.Strategies.STAT_Z_EXIT
        self.invested = 0 # 0 = None, 1 = Long Spread, -1 = Short Spread
        self.invested = 0 # 0 = None, 1 = Long Spread, -1 = Short Spread
        
        # Phase 6: Permissive Mode Override
        if Config.BINANCE_USE_DEMO and getattr(Config.Strategies, 'PERMISSIVE_MODE', False):
            # Lower Z-Entry slightly for more frequency in Demo Competition
            self.z_entry = Config.Strategies.STAT_Z_ENTRY * 0.8 # 20% more permissive
            print(f"🧪 LAB MODE: {self.pair} using Permissive Z-Entry={self.z_entry:.2f}")

    def _check_proactive_expectancy(self, symbol) -> bool:
        """
        Phase 5: Proactive Expectancy Filter.
        Returns False if Expected Value (E) < 0.
        """
        try:
            dh = get_data_handler()
            # Assuming standard trades path - could be configurable
            csv_path = "dashboard/data/trades.csv" 
            trades = dh.load_trades_df(csv_path)
            
            if trades.empty:
                return True # No history -> permissive
                
            # Filter for this strategy/symbol if possible. 
            # For now, simplistic approach: check symbol specific stats.
            sym_trades = trades[trades['symbol'] == symbol]
            if len(sym_trades) < 10:
                print(f"📊 [Expectancy] {symbol}: Insufficient history ({len(sym_trades)}). Learning mode.")
                return True
                
            stats = AnalyticsEngine.calculate_expectancy(sym_trades)
            e_proj = stats.get('expectancy', 0.0)
            friction = AnalyticsEngine.calculate_friction(sym_trades).get('friction_pct', 0.0)
            
            # Penalize E with current Friction if not already accounted
            # Note: calculate_expectancy usually uses Net PnL, so friction is included.
            # But let's be strict: if E is marginally positive but friction is huge, unsafe.
            
            if e_proj <= 0:
                print(f"🛑 [Expectancy] {symbol}: Blocked. E={e_proj:.4f} <= 0 (Friction: {friction:.1f}%)")
                return False
                
            print(f"✅ [Expectancy] {symbol}: Passed. E={e_proj:.4f} > 0")
            return True
            
        except Exception as e:
            logger.error(f"⚠️ Expectancy check error: {e}")
            return True # Fail open to avoid freezing strategy on error

    def calculate_signals(self, event):
        try:
            if event.type == EventType.MARKET:
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

                # Deduplication & Cooldown (Centralized)
                can_process, _ = cooldown_manager.can_trade(y_sym, strategy_id="STATISTICAL")
                if not can_process:
                    return
                # We don't record trade yet, only processing check.
                # Actually, for data processing frequency control, we can just use a local timestamp or let CooldownManager handle a "PROCESSING" key
                
                # Using CooldownManager for frequency control of the PAIR
                # Key: STAT_PROCESS_{y_sym}_{x_sym}
                process_key = f"STAT_PROCESS_{y_sym}_{x_sym}"
                if not cooldown_manager.check_custom_cooldown(process_key, duration_seconds=60): # 1 min check
                     return


                if not cooldown_manager.check_custom_cooldown(process_key, duration_seconds=60): # 1 min check
                     return


                # HURST EXPONENT FILTER (Phase 6)
                # Before heavy processing, check Market Memory.
                # If H > 0.55 -> Persistent (Trending) -> BAD for Mean Reversion.
                # If H < 0.45 -> Anti-persistent -> GOOD for Mean Reversion.
                closes_simple = [b['close'] for b in bars_y] # Use Y (e.g. ETH) as proxy
                if len(closes_simple) >= 100:
                    h_val = StatisticsPro.calculate_hurst_exponent(closes_simple)
                    if h_val > 0.60: # Strong Trend
                        print(f"📉 [Hurst] {y_sym}: Blocked. Market is Trending (H={h_val:.2f} > 0.60).")
                        return
                    elif h_val > 0.53:
                         print(f"⚠️ [Hurst] {y_sym}: Caution via Hurst (H={h_val:.2f}). Expect volatility.")
                
                # BUG #55 FIX: Ensure data alignment and correct Ratio High/Low calculation
                # 1. Align Data by Timestamp (Crucial for correlation/cointegration)
                # We iterate backwards and only keep matching timestamps
                aligned_y = []
                aligned_x = []
                
                # Create dicts for O(1) lookup
                y_dict = {b['datetime']: b for b in bars_y}
                x_dict = {b['datetime']: b for b in bars_x}
                
                common_timestamps = sorted(list(set(y_dict.keys()) & set(x_dict.keys())))
                
                if len(common_timestamps) < self.window:
                    return # Not enough overlapping data
                
                # Reconstruct aligned lists
                bars_y_aligned = [y_dict[ts] for ts in common_timestamps]
                bars_x_aligned = [x_dict[ts] for ts in common_timestamps]
                
                # Extract aligned arrays
                closes_y = np.array([b['close'] for b in bars_y_aligned])
                closes_x = np.array([b['close'] for b in bars_x_aligned])
                highs_y = np.array([b['high'] for b in bars_y_aligned])
                lows_y = np.array([b['low'] for b in bars_y_aligned])
                highs_x = np.array([b['high'] for b in bars_x_aligned])
                lows_x = np.array([b['low'] for b in bars_x_aligned])
                
                # Calculate Ratio
                # ratios = closes_y / closes_x
                ratios = safe_div(closes_y, closes_x)
                
                # Calculate Ratio High/Low correctly
                # Ratio High = Max possible value (High Y / Low X)
                # Ratio Low = Min possible value (Low Y / High X)
                # ratio_highs = highs_y / lows_x
                # ratio_lows = lows_y / highs_x
                ratio_highs = safe_div(highs_y, lows_x)
                ratio_lows = safe_div(lows_y, highs_x)
                
                # Safety: Ensure Low <= High (Floating point issues or bad data could violate this)
                # If Low > High, swap them
                mask = ratio_lows > ratio_highs
                if np.any(mask):
                    ratio_lows[mask], ratio_highs[mask] = ratio_highs[mask], ratio_lows[mask]
                
                # Calculate Ratio ADX
                try:
                    ratio_adx = talib.ADX(ratio_highs, ratio_lows, ratios, timeperiod=14)[-1]
                except Exception:
                    ratio_adx = 0
                
                # Handle NaN
                if np.isnan(ratio_adx):
                    ratio_adx = 0
                
                # Calculate ATR for volatility-based sizing
                # We use the ATR of the asset we are buying (Y or X)
                closes_y = np.array([b['close'] for b in bars_y])
                atr_y = talib.ATR(highs_y, lows_y, closes_y, timeperiod=14)[-1]
                
                closes_x = np.array([b['close'] for b in bars_x])
                atr_x = talib.ATR(highs_x, lows_x, closes_x, timeperiod=14)[-1]

                # Calculate Z-Score with validation for finite values
                # spread = closes_y / closes_x
                spread = safe_div(closes_y, closes_x)
                
                # Validate spread data - filter out NaN/Inf values
                valid_spread = spread[np.isfinite(spread)]
                if len(valid_spread) < 10:
                    return  # Insufficient valid data
                
                mean_spread = np.mean(valid_spread)
                std_spread = np.std(valid_spread)
                
                if std_spread == 0 or not np.isfinite(std_spread):
                    return  # Cannot calculate z-score
                
                z_score = (spread[-1] - mean_spread) / std_spread if np.isfinite(spread[-1]) else 0
                
                # PRINT STATS TO TERMINAL (User Request)
                print(f"📊 Stat Strategy {self.pair}: Z-Score={z_score:.2f} Ratio_ADX={ratio_adx:.1f}")

                # Trading Logic
                # Long Spread = Buy Y, Sell X (Expect ratio to go up)
                # Short Spread = Sell Y, Buy X (Expect ratio to go down)
                
                # Filter: Don't mean revert if Trend is too strong (ADX > 30)
                # DYNAMIC Z-SCORE: If trend is strong, require higher Z-score to enter (don't catch falling knife)
                effective_z_entry = self.z_entry
                
                if ratio_adx > 25:
                    effective_z_entry = self.z_entry * 1.5 # Require 2.25 sigma if trending
                    print(f"  ⚠️ Strong Trend in Spread (ADX={ratio_adx:.1f}). Widening bands to {effective_z_entry:.2f}")
                
                if ratio_adx > 40:
                     # Extreme trend, block mean reversion unless extreme extension
                     effective_z_entry = 4.0
                     
                # Check Portfolio for actual positions (Source of Truth)

                # Check Portfolio for actual positions (Source of Truth)
                # We need to know if we are currently holding the pair
                # This is more robust than a local 'invested' flag
                if not self.portfolio:
                    print("❌ STAT STRATEGY ERROR: Portfolio not initialized!")
                    return
                    
                pos_y = self.portfolio.positions.get(y_sym, {'quantity': 0})
                pos_x = self.portfolio.positions.get(x_sym, {'quantity': 0})
                
                qty_y = pos_y['quantity']
                qty_x = pos_x['quantity']
                
                # Determine current state based on portfolio
                current_state = 0
                is_broken_state = False
                
                if qty_y > 0 and qty_x < 0:
                    current_state = 1 # Long Spread (Long Y, Short X)
                elif qty_y < 0 and qty_x > 0:
                    current_state = -1 # Short Spread (Short Y, Long X)
                elif qty_y != 0 or qty_x != 0:
                    # One is zero, the other is not -> BROKEN STATE (Naked position)
                    # Or both same direction (unlikely but possible error)
                    is_broken_state = True
                    print(f"⚠️  STAT STRATEGY BROKEN STATE: {y_sym}={qty_y}, {x_sym}={qty_x}")
                
                # Update local state to match reality
                self.invested = current_state

                # EMERGENCY HANDLING FOR BROKEN STATE
                if is_broken_state:
                    # We have a naked position. We should close it to reset.
                    # This prevents the strategy from thinking it's flat and entering again.
                    if qty_y != 0:
                        print(f"  🚑 Closing naked leg {y_sym}")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(2, y_sym, signal_timestamp, SignalType.EXIT, strength=1.0))
                    if qty_x != 0:
                        print(f"  🚑 Closing naked leg {x_sym}")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(2, x_sym, signal_timestamp, SignalType.EXIT, strength=1.0))
                    return # Stop processing

                if self.invested == 0:
                    if z_score < -effective_z_entry:
                        # Check Trend for Y (ETH)
                        trend_y = self._get_1h_trend(y_sym)
                        if trend_y == 'DOWN':
                            print(f"  >> Stat Skip {y_sym}: 1h Trend is DOWN")
                        else:
                            # DYNAMIC STRENGTH: Scale based on Z-Score magnitude
                            z_diff = abs(z_score) - self.z_entry
                            strength = min(1.0, 0.5 + (z_diff * 0.2))
                            
                            # PROACTIVE EXPECTANCY CHECK (Phase 5)
                            if not self._check_proactive_expectancy(y_sym):
                                return
                                
                            print(f"ENTRY LONG SPREAD: Buy {y_sym}, Short {x_sym} (Z={z_score:.2f}, Strength={strength:.2f}, 1h Trend: {trend_y})")
                            
                            from datetime import datetime, timezone
                            signal_timestamp = datetime.now(timezone.utc)
                            
                            self.events_queue.put(SignalEvent(2, y_sym, signal_timestamp, SignalType.LONG, strength=strength, atr=atr_y))
                            self.events_queue.put(SignalEvent(2, x_sym, signal_timestamp, SignalType.SHORT, strength=strength, atr=atr_x))
                            # self.invested = 1 # Wait for fill
                            
                    elif z_score > effective_z_entry:
                        # Check Trend for X (BTC)
                        trend_x = self._get_1h_trend(x_sym)
                        if trend_x == 'DOWN':
                            print(f"  >> Stat Skip {x_sym}: 1h Trend is DOWN")
                        else:
                            # DYNAMIC STRENGTH
                            z_diff = abs(z_score) - self.z_entry
                            strength = min(1.0, 0.5 + (z_diff * 0.2))
                            
                            # PROACTIVE EXPECTANCY CHECK (Phase 5)
                            if not self._check_proactive_expectancy(y_sym): # Check primary driver
                                return

                            print(f"ENTRY SHORT SPREAD: Short {y_sym}, Buy {x_sym} (Z={z_score:.2f}, Strength={strength:.2f}, 1h Trend: {trend_x})")
                            
                            from datetime import datetime, timezone
                            signal_timestamp = datetime.now(timezone.utc)
                            
                            self.events_queue.put(SignalEvent(2, y_sym, signal_timestamp, SignalType.SHORT, strength=strength, atr=atr_y))
                            self.events_queue.put(SignalEvent(2, x_sym, signal_timestamp, SignalType.LONG, strength=strength, atr=atr_x))
                            # self.invested = -1 # Wait for fill

                elif self.invested == 1:
                    # Exit Long Spread when Z-score returns to mean
                    if z_score >= -self.z_exit:
                        print(f"EXIT LONG SPREAD (Z={z_score:.2f})")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(2, y_sym, signal_timestamp, SignalType.EXIT, strength=1.0))
                        self.events_queue.put(SignalEvent(2, x_sym, signal_timestamp, SignalType.EXIT, strength=1.0))

                elif self.invested == -1:
                    # Exit Short Spread when Z-score returns to mean
                    if z_score <= self.z_exit:
                        print(f"EXIT SHORT SPREAD (Z={z_score:.2f})")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(2, y_sym, signal_timestamp, SignalType.EXIT, strength=1.0))
                        self.events_queue.put(SignalEvent(2, x_sym, signal_timestamp, SignalType.EXIT, strength=1.0))
        except Exception as e:
            print(f"⚠️  Statistical Strategy Error: {e}")

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

