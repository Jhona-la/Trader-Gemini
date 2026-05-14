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
from core.neural_bridge import neural_bridge

class StatisticalStrategy(Strategy):
    """
    Pairs Trading Strategy based on Cointegration / Mean Reversion of the Spread.
    """
    def __init__(self, data_provider, events_queue, portfolio=None, pair=('ETH/USDT', 'BTC/USDT'), horizon="SCALPING", priority: int = 1):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.portfolio = portfolio # Needed for state tracking
        self.pair = pair # Tuple of two symbols (Y, X) where Spread = Y - beta*X or Ratio = Y/X
        self.horizon = horizon
        self.priority = priority
        self.window = Config.Strategies.STAT_WINDOW
        self.long_window = self.window * 10 # Phase 7+: 200 bars for long-term baseline
        self.z_base = Config.Strategies.STAT_Z_ENTRY
        self.z_entry = Config.Strategies.STAT_Z_ENTRY
        self.z_exit = Config.Strategies.STAT_Z_EXIT
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
            if event.type != EventType.MARKET:
                return

            # OMNI-SYNC: Iterate over all symbols in the basket vs BTC
            # This allows the statistical strategy to capture mean reversion in all "Gems"
            target_symbols = [s for s in self.data_provider.symbol_list if 'BTC' not in s]
            x_sym = 'BTC/USDT'
            
            for y_sym in target_symbols:
                # Deduplication & Cooldown per pair
                process_key = f"STAT_PROCESS_{y_sym}_{x_sym}"
                if not cooldown_manager.check_custom_cooldown(process_key, duration_seconds=60):
                    continue

                # Get latest bars for both assets
                try:
                    bars_y = self.data_provider.get_latest_bars(y_sym, n=self.window)
                    bars_x = self.data_provider.get_latest_bars(x_sym, n=self.window)
                except KeyError:
                    continue 

                if bars_y is None or bars_x is None or len(bars_y) < self.window or len(bars_x) < self.window:
                    continue

                # ═══════════════════════════════════════════════════════
                # FORENSIC-DCA FIX: REMOVED TRIPLICATED COOLDOWN CHECK
                # ANTES: 3 cooldown checks redundantes (L93, L108-110, L116-122)
                #   que usaban `return` (mata TODO el loop) en vez de `continue`.
                # AHORA: Un solo check en L93 con `continue` (solo salta este par).
                # POR QUÉ: El primer check (L93) ya controla 60s cooldown por par.
                #   Los dos siguientes eran duplicados que causaban `return` prematuro,
                #   matando el análisis de TODOS los pares restantes.
                # ═══════════════════════════════════════════════════════

                # HURST EXPONENT FILTER (Phase 6/7)
                # Before heavy processing, check Market Memory.
                # If H > 0.55 -> Persistent (Trending) -> BAD for Mean Reversion.
                # If H < 0.45 -> Anti-persistent -> GOOD for Mean Reversion.
                closes_simple = [b['close'] for b in bars_y] # Use Y (e.g. ETH) as proxy
                h_val = 0.5 # Default Neutral
                if len(closes_simple) >= 100:
                    h_val = StatisticsPro.calculate_hurst_exponent(closes_simple)
                    if h_val > 0.70: # Extreme Trend
                        print(f"📉 [Hurst] {y_sym}: Blocked. Extreme Trend (H={h_val:.2f} > 0.70).")
                        return
                    elif h_val > 0.55:
                         print(f"⚠️ [Hurst] {y_sym}: Caution (H={h_val:.2f}). Market trending.")

                # ═══════════════════════════════════════════════════
                # FORENSIC-V42: REGIME BLOCK FOR MEAN REVERSION
                # QUÉ: Bloquear reversión a la media en régimen TRENDING extremo.
                # POR QUÉ: Operar contra la tendencia destruye cuentas de $13.
                # ═══════════════════════════════════════════════════
                market_regime = "UNKNOWN"
                if self.portfolio and hasattr(self.portfolio, 'global_regime_data'):
                    regime_meta = self.portfolio.global_regime_data
                    market_regime = regime_meta.get('symbol_regimes', {}).get(y_sym, regime_meta.get('sentiment', 'UNKNOWN'))
                if market_regime == "TRENDING" and h_val > 0.60:
                    logger.info(f"🛑 [STAT REGIME BLOCK] {y_sym}: Mean reversion blocked in TRENDING regime (H={h_val:.2f}).")
                    return
                
                # BUG #55 FIX: Ensure data alignment and correct Ratio High/Low calculation
                # 1. Align Data by Timestamp (Crucial for correlation/cointegration)
                # We iterate backwards and only keep matching timestamps
                aligned_y = []
                aligned_x = []
                
                # Create dicts for O(1) lookup
                # FORENSIC-V42 FIX: Use 'timestamp' (backtest) with fallback to 'datetime' (production)
                def _get_ts(b):
                    if 'datetime' in b.dtype.names if hasattr(b, 'dtype') else 'datetime' in b:
                        return b['datetime']
                    return b['timestamp']
                
                try:
                    y_dict = {_get_ts(b): b for b in bars_y}
                    x_dict = {_get_ts(b): b for b in bars_x}
                except (KeyError, ValueError):
                    # Structured array: use index-based alignment instead
                    min_len = min(len(bars_y), len(bars_x))
                    bars_y_aligned = bars_y[-min_len:]
                    bars_x_aligned = bars_x[-min_len:]
                    y_dict = None
                    x_dict = None
                
                if y_dict is not None and x_dict is not None:
                    common_timestamps = sorted(list(set(y_dict.keys()) & set(x_dict.keys())))
                    
                    if len(common_timestamps) < self.window:
                        return # Not enough overlapping data
                    
                    # Reconstruct aligned lists
                    bars_y_aligned = [y_dict[ts] for ts in common_timestamps]
                    bars_x_aligned = [x_dict[ts] for ts in common_timestamps]
                
                # Extract aligned arrays — F6: Cast float32→float64 for talib
                closes_y = np.array([b['close'] for b in bars_y_aligned], dtype=np.float64)
                closes_x = np.array([b['close'] for b in bars_x_aligned], dtype=np.float64)
                highs_y = np.array([b['high'] for b in bars_y_aligned], dtype=np.float64)
                lows_y = np.array([b['low'] for b in bars_y_aligned], dtype=np.float64)
                highs_x = np.array([b['high'] for b in bars_x_aligned], dtype=np.float64)
                lows_x = np.array([b['low'] for b in bars_x_aligned], dtype=np.float64)
                
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
                closes_y = np.array([b['close'] for b in bars_y], dtype=np.float64)  # F6
                atr_y = talib.ATR(highs_y, lows_y, closes_y, timeperiod=14)[-1]
                
                closes_x = np.array([b['close'] for b in bars_x], dtype=np.float64)  # F6
                atr_x = talib.ATR(highs_x, lows_x, closes_x, timeperiod=14)[-1]

                # Calculate Rolling OLS Beta (Dynamic Hedge Ratio)
                # Phase 4 Math Upgrade
                
                # Use log prices for OLS to capture percentage relationships
                log_y = np.log(closes_y)
                log_x = np.log(closes_x)
                
                # Rolling OLS to get Beta (Hedge Ratio)
                # Phase 6 Upgrade: RANSAC (Robust Regulation)
                # RANSAC ignores outliers (Flash crashes) for a pure structural relationship
                beta, alpha = StatisticsPro.ransac_regression(log_y, log_x, window=min(50, len(log_y)))
                
                # Calculate Spread: Spread = log(Y) - beta * log(X)
                # This creates a stationary series (cointegration)
                spread = log_y - beta * log_x
                
                # Calculate Half-Life for dynamic window adjustment (Phase 4)
                half_life = StatisticsPro.calculate_half_life(spread)
                
                # Calculate Z-Score
                # Dynamic Window based on Half-Life (2x Half-Life is common for mean reversion)
                # But keep it bounded (20 to 100)
                if half_life > 0:
                    z_window = int(max(20, min(100, half_life * 2)))
                else:
                    z_window = self.window

                # Rolling Z-Score on spread
                roll_mean = pd.Series(spread).rolling(window=z_window).mean().values
                roll_std = pd.Series(spread).rolling(window=z_window).std().values
                
                # Current Z
                if roll_std[-1] > 0 and np.isfinite(roll_std[-1]):
                    z_score = (spread[-1] - roll_mean[-1]) / roll_std[-1]
                else:
                    z_score = 0
                
                # Log critical math stats
                # print(f"DEBUG STATS: Beta(RANSAC)={beta:.4f} HL={half_life:.1f} win={z_window} Z={z_score:.2f}")
                
                # Phase 6: Export Stats to Portfolio (Dashboard Propagation)
                if self.portfolio:
                    self.portfolio.update_math_stats({
                        'beta': float(beta), # RANSAC Beta
                        'half_life': float(half_life),
                        'z_score': float(z_score),
                        # Hurst is calculated in MarketRegime, but we can add proxy here if needed
                        # Ideally strategies report their own internal metrics
                    })
                
                # Phase 6: Proactive Gatekeeper (Expectancy)
                # Check if this pair has positive expectancy locally
                if self.portfolio: # Only if portfolio and data available
                    # Mock fetching trades for this pair (Logic would be in DataHandler usually)
                    # For now, we assume global expectancy check is done in _check_proactive_expectancy
                    pass # Done in entry logic
                
                # PHASE 7+: ADAPTIVE Z-SCORE & VOLATILITY SYNC
                # Calculate short-term spread std FIRST (was missing, caused NameError)
                std_spread = roll_std[-1] if np.isfinite(roll_std[-1]) and roll_std[-1] > 0 else np.std(spread)

                # 1. Long-term baseline (sigma_long)
                try:
                    bars_y_long = self.data_provider.get_latest_bars(y_sym, n=self.long_window)
                    bars_x_long = self.data_provider.get_latest_bars(x_sym, n=self.long_window)
                    if bars_y_long is not None and bars_x_long is not None and len(bars_y_long) >= 100:
                        y_long = np.array([b['close'] for b in bars_y_long], dtype=np.float64)  # F6
                        x_long = np.array([b['close'] for b in bars_x_long], dtype=np.float64)  # F6
                        spread_long = safe_div(y_long, x_long)
                        std_long = np.std(spread_long[np.isfinite(spread_long)])
                    else:
                        std_long = std_spread
                except:
                    std_long = std_spread

                # 2. Calculate Volatility Ratio (sigma_short / sigma_long)
                vol_ratio = std_spread / std_long if std_long > 0 else 1.0
                
                # 3. Adaptive Threshold Formula: Z_adapt = Z_base * Vol_Ratio
                # If volatility is high, we require a larger Z-Score (Don't buy the flush)
                adaptive_z = self.z_base * vol_ratio
                
                # 4. Integrate Hurst Penalty
                # If H > 0.60 (Strong Trend), we punish the threshold further
                if h_val > 0.60:
                    adaptive_z *= 1.5 # 50% extra penalty for trending markets
                elif h_val > 0.55:
                    adaptive_z *= 1.25 # 25% penalty
                
                # 5. Flash Crash & Micro-Capital Protection
                # If volatility spikes > 3x normal, increase z to a level that effectively bans trading
                # FORENSIC-V47: Added sanity check — if std_long < 1e-6 (no baseline yet), vol_ratio is meaningless
                if vol_ratio > 3.0 and std_long > 1e-6:
                    adaptive_z *= 2.0
                    if vol_ratio < 50.0:  # Only print for real spikes, not near-zero baseline artifacts
                        print(f"🚨 [FLASH CRASH ALERT] Volatility {vol_ratio:.1f}x above baseline. Shielding $13.50.")

                # Cap adaptive Z to avoid extreme values but keep it high
                effective_z_entry = min(5.0, max(self.z_base, adaptive_z))

                # PRINT STATS TO TERMINAL (User Request)
                # print(f"📊 Stat Strategy {y_sym}/{x_sym}: Z-Score={z_score:.2f} (Target={effective_z_entry:.2f}) Vol_Ratio={vol_ratio:.2f}x H={h_val:.2f}")

                # Trading Logic
                # Long Spread = Buy Y, Sell X (Expect ratio to go up)
                # Short Spread = Sell Y, Buy X (Expect ratio to go down)
                
                # Filter: Don't mean revert if Trend is too strong (ADX > 30)
                # DYNAMIC Z-SCORE: If trend is strong, require higher Z-score to enter (don't catch falling knife)
                if ratio_adx > 25:
                    effective_z_entry = max(effective_z_entry, self.z_base * 1.5) 
                    print(f"  ⚠️ Strong Trend in Spread (ADX={ratio_adx:.1f}). Clamping Z at {effective_z_entry:.2f}")
                
                if ratio_adx > 40:
                     # Extreme trend, block mean reversion unless extreme extension
                     effective_z_entry = 4.5
                     
                # Check Portfolio for actual positions (Source of Truth)

                # Check Portfolio for actual positions (Source of Truth)
                # We need to know if we are currently holding the pair
                # This is more robust than a local 'invested' flag
                if not self.portfolio:
                    print("❌ STAT STRATEGY ERROR: Portfolio not initialized!")
                    return
                    
                pos_y = {}
                pos_x = {}
                if hasattr(self.portfolio, 'get_horizon_position'):
                    pos_y = self.portfolio.get_horizon_position(y_sym, self.horizon) or {}
                    pos_x = self.portfolio.get_horizon_position(x_sym, self.horizon) or {}
                else:
                    pos_y = self.portfolio.positions.get(y_sym, {})
                    pos_x = self.portfolio.positions.get(x_sym, {})

                # Ensure the positions actually belong to the Statistical Strategy
                strat_id = getattr(self, 'strategy_id', 'STAT_ARB')
                if pos_y.get('quantity', 0) != 0 and pos_y.get('strategy_id') != strat_id:
                    pos_y = {}
                if pos_x.get('quantity', 0) != 0 and pos_x.get('strategy_id') != strat_id:
                    pos_x = {}
                    
                qty_y = pos_y.get('quantity', 0)
                qty_x = pos_x.get('quantity', 0)
                
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
                
                # Phase 8: Neural Bridge Publication (Broadcasting Conviction)
                neural_bridge.publish_insight(
                    strategy_id="STAT_SPREAD",
                    symbol=y_sym, # Primary symbol for the pair
                    insight={
                        'confidence': min(1.0, abs(z_score) / effective_z_entry),
                        'direction': 'LONG' if z_score < 0 else 'SHORT', # Z < 0 means buy Y (LONG)
                        'z_score': z_score,
                        'vol_ratio': vol_ratio
                    }
                )

                # EMERGENCY HANDLING FOR BROKEN STATE
                if is_broken_state:
                    # We have a naked position. We should close it to reset.
                    # This prevents the strategy from thinking it's flat and entering again.
                    if qty_y != 0:
                        print(f"  🚑 Closing naked leg {y_sym}")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=y_sym, datetime=signal_timestamp, signal_type=SignalType.EXIT, strength=1.0, horizon=self.horizon, priority=self.priority))
                    if qty_x != 0:
                        print(f"  🚑 Closing naked leg {x_sym}")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=x_sym, datetime=signal_timestamp, signal_type=SignalType.EXIT, strength=1.0, horizon=self.horizon, priority=self.priority))
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
                                
                            # ── SOPHIA-INTELLIGENCE: Pre-trade XAI Analysis ──
                            sophia_report_dict_y = {}
                            if hasattr(self, 'sophia') and self.sophia:
                                sophia_report = self.sophia.analyze(
                                    symbol=y_sym,
                                    direction="LONG",
                                    signal_strength=strength,
                                    setups={'z_score': z_score, 'pair': x_sym},
                                    confluence_score=1.0,
                                    tp_pct=Config.Strategies.SWING_PARAMS['tp_pct'] if self.horizon == 'SWING' else Config.Strategies.SCALPING_PARAMS['tp_pct'],
                                    sl_pct=Config.Strategies.SWING_PARAMS['sl_pct'] if self.horizon == 'SWING' else Config.Strategies.SCALPING_PARAMS['sl_pct'],
                                    returns=None,
                                    ttl_seconds=3600.0 if self.horizon == 'SWING' else 300.0,
                                    regime="RANGING", # Pairs trading usually implies ranging/mean rev
                                )
                    
                                # FORENSIC-V42: DYNAMIC EXACTITUDE THRESHOLD
                                stat_veto = 0.55  # Mean reversion is more tolerant
                                if sophia_report.win_probability < stat_veto:
                                    logger.info(f"🛑 [SOPHIA VETO] {y_sym} Stat Signal Blocked. Exactitude ({sophia_report.win_probability*100:.1f}%) < {stat_veto*100:.0f}%.")
                                    return
                                sophia_report_dict_y = sophia_report.to_dict()

                            print(f"ENTRY LONG SPREAD: Buy {y_sym}, Short {x_sym} (Z={z_score:.2f}, Strength={strength:.2f}, 1h Trend: {trend_y})")
                            
                            from datetime import datetime, timezone
                            signal_timestamp = datetime.now(timezone.utc)
                            
                            self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=y_sym, datetime=signal_timestamp, signal_type=SignalType.LONG, strength=strength, atr=atr_y, horizon=self.horizon, priority=self.priority, metadata={'sophia': sophia_report_dict_y}))
                            self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=x_sym, datetime=signal_timestamp, signal_type=SignalType.SHORT, strength=strength, atr=atr_x, horizon=self.horizon, priority=self.priority, metadata={'sophia': sophia_report_dict_y}))
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

                            # ── SOPHIA-INTELLIGENCE: Pre-trade XAI Analysis ──
                            sophia_report_dict_y = {}
                            if hasattr(self, 'sophia') and self.sophia:
                                sophia_report = self.sophia.analyze(
                                    symbol=y_sym,
                                    direction="SHORT",
                                    signal_strength=strength,
                                    setups={'z_score': z_score, 'pair': x_sym},
                                    confluence_score=1.0,
                                    tp_pct=Config.Strategies.SWING_PARAMS['tp_pct'] if self.horizon == 'SWING' else Config.Strategies.SCALPING_PARAMS['tp_pct'],
                                    sl_pct=Config.Strategies.SWING_PARAMS['sl_pct'] if self.horizon == 'SWING' else Config.Strategies.SCALPING_PARAMS['sl_pct'],
                                    returns=None,
                                    ttl_seconds=3600.0 if self.horizon == 'SWING' else 300.0,
                                    regime="RANGING", # Pairs trading usually implies ranging/mean rev
                                )
                    
                                # FORENSIC-V42: DYNAMIC EXACTITUDE THRESHOLD
                                stat_veto = 0.55  # Mean reversion is more tolerant
                                if sophia_report.win_probability < stat_veto:
                                    logger.info(f"🛑 [SOPHIA VETO] {y_sym} Stat Signal Blocked. Exactitude ({sophia_report.win_probability*100:.1f}%) < {stat_veto*100:.0f}%.")
                                    return
                                sophia_report_dict_y = sophia_report.to_dict()

                            print(f"ENTRY SHORT SPREAD: Short {y_sym}, Buy {x_sym} (Z={z_score:.2f}, Strength={strength:.2f}, 1h Trend: {trend_x})")
                            
                            from datetime import datetime, timezone
                            signal_timestamp = datetime.now(timezone.utc)
                            
                            self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=y_sym, datetime=signal_timestamp, signal_type=SignalType.SHORT, strength=strength, atr=atr_y, horizon=self.horizon, priority=self.priority, metadata={'sophia': sophia_report_dict_y}))
                            self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=x_sym, datetime=signal_timestamp, signal_type=SignalType.LONG, strength=strength, atr=atr_x, horizon=self.horizon, priority=self.priority, metadata={'sophia': sophia_report_dict_y}))
                            # self.invested = -1 # Wait for fill

                elif self.invested == 1:
                    # Exit Long Spread when Z-score returns to mean
                    if z_score >= -self.z_exit:
                        print(f"EXIT LONG SPREAD (Z={z_score:.2f})")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=y_sym, datetime=signal_timestamp, signal_type=SignalType.EXIT, strength=1.0, horizon=self.horizon, priority=self.priority))
                        self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=x_sym, datetime=signal_timestamp, signal_type=SignalType.EXIT, strength=1.0, horizon=self.horizon, priority=self.priority))

                elif self.invested == -1:
                    # Exit Short Spread when Z-score returns to mean
                    if z_score <= self.z_exit:
                        print(f"EXIT SHORT SPREAD (Z={z_score:.2f})")
                        
                        from datetime import datetime, timezone
                        signal_timestamp = datetime.now(timezone.utc)
                        
                        self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=y_sym, datetime=signal_timestamp, signal_type=SignalType.EXIT, strength=1.0, horizon=self.horizon, priority=self.priority))
                        self.events_queue.put(SignalEvent(strategy_id="STAT_V1", symbol=x_sym, datetime=signal_timestamp, signal_type=SignalType.EXIT, strength=1.0, horizon=self.horizon, priority=self.priority))
        except Exception as e:
            print(f"⚠️  Statistical Strategy Error: {e}")

    def _get_1h_trend(self, symbol):
        """
        Helper to get 1h trend using CLOSED candles.
        """
        try:
            bars_1h = self.data_provider.get_latest_bars_1h(symbol, n=210)
            if bars_1h is not None and len(bars_1h) >= 200:
                closes_1h = np.array([b['close'] for b in bars_1h[:-1]], dtype=np.float64)  # F6: float64 for talib
                if len(closes_1h) >= 200:
                    ema_50 = talib.EMA(closes_1h, timeperiod=50)[-1]
                    ema_200 = talib.EMA(closes_1h, timeperiod=200)[-1]
                    return 'UP' if ema_50 > ema_200 else 'DOWN'
        except:
            pass
        return 'NEUTRAL'

