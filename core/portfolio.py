
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import pandas as pd

from core.enums import TradeDirection, TradeStatus, EventType, OrderSide, OrderType
from core.reward_system import TradeOutcome
from utils.logger import logger
from utils.notifier import Notifier
from utils.data_manager import DatabaseHandler, safe_append_csv, safe_read_csv
from utils.atomic_guard import AtomicGuard
from core.state_manager import AtomicStateManager # Phase 27
from sophia.post_mortem import PostMortemComparator  # SOPHIA-INTELLIGENCE Protocol
from sophia.nemesis import NemesisEngine  # NÉMESIS-RETROSPECCIÓN Protocol
from utils.axioma_math import PrecisionAuditor  # CRITERIO-AXIOMA Protocol

from typing import Dict, Any

from config import Config
from utils.debug_tracer import trace_execution
from decimal import Decimal, getcontext

class Portfolio:
    def __init__(self, initial_capital: float = 10000.0, 
                 csv_path: str = "dashboard/data/trades.csv", 
                 status_path: str = "dashboard/data/status.csv", 
                 auto_save: bool = True):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.pending_cash = 0.0  # Cash reserved for pending orders
        self.used_margin = 0.0   # Margin locked in Futures positions
        
        # [PRECISION-AXIOMA]
        self.precision_drift_accumulated = Decimal('0.0')
        getcontext().prec = 28 # Satoshi-level precision for drift auditing
        
        self.positions = {} # Symbol -> {'quantity': 0, 'avg_price': 0, 'current_price': 0}
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0  # CRITERIO-AXIOMA: Explicit fee tracking
        
        # STRATEGY ATTRIBUTION: Track PnL per strategy
        # Format: {strategy_id: {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0}}
        self.strategy_performance = {}
        
        # PHASE 14: Dynamic Kelly Criterion Tracking
        self.kelly_trades_history = []  # List of dicts: {'pnl': float, 'is_win': bool}
        self.kelly_winrate = 0.0
        self.kelly_payoff_ratio = 1.0
        self.KELLY_WINDOW = 20
        
        self.csv_path = csv_path
        self.status_path = status_path
        self.auto_save = auto_save
        
        # Phase 50: Atomic Metal-Core Protection
        self.guard = AtomicGuard()
        
        # Isolated state caches (Fast Access)
        self._equity_cache = initial_capital
        self._last_snapshot = None
        self._last_snapshot_ts = 0
        
        self.io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PortfolioIO")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Initialize CSVs
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=['datetime', 'symbol', 'type', 'direction', 'quantity', 'price', 'details'])
            df.to_csv(self.csv_path, index=False)
            
        # Create initial status file
        self.save_status()
        
        # Initialize Database
        self.db = DatabaseHandler()
        
        # Phase 6: Math Stats Tracking
        self.math_stats = {
            'hurst': 0.5,
            'beta': 1.0,
            'half_life': 0,
            'last_update': None
        }
        # Phase 7: Meta-Brain Stats
        self.strategy_rankings = {}
        
        # SOPHIA-INTELLIGENCE: Post-Mortem Calibration Tracker
        self.sophia_post_mortem = PostMortemComparator(rolling_window=100)
        
        # NÉMESIS-RETROSPECCIÓN: Deep Post-Mortem Autopsy Engine
        self.nemesis_engine = NemesisEngine()
        
    def get_atomic_snapshot(self) -> Dict[str, Any]:
        """
        Returns a thread-safe deep copy of the portfolio state.
        Guarantees internal consistency (e.g. Equity = Cash + PnL).
        """
        self.guard.acquire()
        try:
            # Snapshot critical fields
            snapshot = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cash': float(self.current_cash),
                'realized_pnl': float(self.realized_pnl),
                'used_margin': float(self.used_margin),
                'pending_cash': float(self.pending_cash),
                'equity': float(self._equity_cache),
                'positions': {
                    sym: pos.copy() for sym, pos in self.positions.items()
                },
                'math_stats': self.math_stats.copy()
            }
            return snapshot
        finally:
            self.guard.release()

    def update_math_stats(self, stats: Dict[str, Any]) -> None:
        """Update live mathematical statistics from strategies."""
        if not stats: return
        self.guard.acquire()
        try:
            self.math_stats.update(stats)
        finally:
            self.guard.release()
        # Optional: Auto-save if critical diff? 
        # For now, rely on standard loop save
        
    def restore_state_from_db(self):
        """
        Restore portfolio state (positions) from SQLite database.
        Used for crash recovery.
        """
        try:
            # Restore Positions
            db_positions = self.db.get_open_positions()
            if db_positions:
                self.positions = db_positions
                logger.info(f"🔄 RESTORED {len(self.positions)} active positions from DB.")
                for sym, pos in self.positions.items():
                    logger.info(f"   - {sym}: {pos['quantity']} @ ${pos['entry_price']:.4f}")
                    # Map entry_price to avg_price for internal consistency
                    if 'avg_price' not in pos:
                        pos['avg_price'] = pos['entry_price']
            else:
                logger.info("✅ No active positions found in DB.")
                
            return True
        except Exception as e:
            logger.error(f"⚠️  Failed to restore portfolio state from DB: {e}")
            return False
        
    def load_portfolio_state(self, state_path):
        """
        Load portfolio state (positions, cash, pnl) from a JSON file.
        Used for crash recovery.
        """
        if not os.path.exists(state_path):
            return False
            
        try:
            from utils.fast_json import FastJson
            data = FastJson.load_from_file(state_path)
            if data is None: return False
                
            if data.get('status') == 'OFFLINE':
                pass
                
            # Restore Cash & PnL
            self.current_cash = data.get('cash', self.initial_capital)
            self.realized_pnl = data.get('realized_pnl', 0.0)
            self.used_margin = data.get('used_margin', 0.0) # Restore margin
            
            # Restore Positions
            loaded_positions = data.get('positions', {})
            if loaded_positions:
                self.positions = loaded_positions
                print(f"🔄 RESTORED {len(self.positions)} active positions from previous session.")
                for sym, pos in self.positions.items():
                    print(f"   - {sym}: {pos['quantity']} @ ${pos['avg_price']:.4f}")
            else:
                print("✅ No active positions to restore.")
                
            return True
        except Exception as e:
            print(f"⚠️  Failed to restore portfolio state: {e}")
            return False
            
    def check_systemic_risk(self):
        """
        [PHASE 20] Calculates Fleet Beta / Correlation (Immunology).
        If Correlation > 0.9, marks regime as SYSTEMIC_COLLAPSE.
        """
        try:
            dh = get_data_handler()
            if not dh: return
            
            symbols = dh.symbol_list
            if len(symbols) < 3: return
            
            # Fetch returns for all symbols
            returns_map = {}
            for s in symbols:
                bars = dh.get_latest_bars(s, n=50) # Last 50m
                if bars is not None and len(bars) > 40:
                     # Access structured array 'close' via safe getter if possible, 
                     # but here we assume numpy array access from loader
                     # Loader returns structured array with 'close' field
                     closes = bars['close']
                     # Calculate returns
                     rets = np.diff(closes) / closes[:-1]
                     returns_map[s] = rets
            
            if not returns_map: return

            # Pad/Align (simplified: assumes sync or just takes min length)
            min_len = min(len(r) for r in returns_map.values())
            if min_len < 30: return
            
            # Slice to same length
            data = {s: r[-min_len:] for s, r in returns_map.items()}
            df = pd.DataFrame(data)
            
            corr_matrix = df.corr()
            # Mean of off-diagonal elements ideally, but full mean is okay proxy
            avg_corr = corr_matrix.mean().mean()
            
            # Store in Stats
            self.math_stats['fleet_corr'] = avg_corr
            self.math_stats['last_update'] = time.time()
            
            if avg_corr > 0.9:
                logger.warning(f"☢️ [IMMUNOLOGY] HIGH SYSTEMIC RISK! Avg Corr: {avg_corr:.3f}")
                self.math_stats['systemic_risk'] = True
            else:
                 self.math_stats['systemic_risk'] = False
                 
        except Exception as e:
            logger.error(f"Systemic Risk Check Failed: {e}")
    
    @trace_execution
    def get_available_cash(self):
        """Return cash available for trading (Total Cash - Margin Used - Pending)."""
        self.guard.acquire()
        try:
            # In Futures: Cash is collateral. Available = Cash - Margin - Pending
            # In Spot: Cash is currency. Available = Cash - Pending (Margin is N/A)
            if Config.BINANCE_USE_FUTURES:
                return self.current_cash - self.used_margin - self.pending_cash
            else:
                return self.current_cash - self.pending_cash
        finally:
            self.guard.release()

    @property
    def unrealized_pnl(self):
        """
        Calculate total unrealized PnL from all open positions using Atomic Guard.
        """
        pnl = 0.0
        self.guard.acquire()
        try:
            for symbol, pos in self.positions.items():
                qty = pos['quantity']
                if qty != 0:
                    avg_price = pos['avg_price']
                    current_price = pos.get('current_price', avg_price)
                    pnl += (current_price - avg_price) * qty
            return pnl
        finally:
            self.guard.release()

    def get_total_equity(self):
        """
        Return Total Equity = Cash + Unrealized PnL.
        SUPREMO-V3: Cached for O(1) read access.
        """
        # Periodic update of cache happens in update_market_price
        return self._equity_cache

    def _refresh_equity_cache(self):
        """Internal heavy calculation of equity."""
        self.guard.acquire()
        try:
            equity = self.current_cash
            for symbol, pos in self.positions.items():
                qty = pos['quantity']
                if qty != 0:
                    avg_price = pos['avg_price']
                    current_price = pos.get('current_price', avg_price)
                    equity += (current_price - avg_price) * qty
            
            self._equity_cache = equity
            return equity
        finally:
            self.guard.release()
    
    @trace_execution
    def reserve_cash(self, amount):
        """Reserve cash for a pending order. Returns True if successful."""
        self.guard.acquire()
        try:
            # Recalculate locally to be safe with lock
            if Config.BINANCE_USE_FUTURES:
                avail = self.current_cash - self.used_margin - self.pending_cash
            else:
                avail = self.current_cash - self.pending_cash
                
            if avail >= amount:
                self.pending_cash += amount
                return True
            return False
        finally:
            self.guard.release()
    
    def release_cash(self, amount):
        """Release reserved cash (order failed/canceled)."""
        self.guard.acquire()
        try:
            self.pending_cash = max(0, self.pending_cash - amount)
        finally:
            self.guard.release()

    def update_timeindex(self, event):
        """
        Update current market prices for all positions.
        """
        if event.type == EventType.MARKET:
            pass 
            
    def update_market_price(self, symbol, price):
        """
        Helper to update current price of a symbol for PnL calculation.
        Updates HWM for LONG positions, LWM for SHORT positions.
        """
        if price <= 0:
            logger.warning(f"Portfolio: Ignoring invalid price for {symbol}: {price}")
            return

        self.guard.acquire()
        try:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0, 
                    'avg_price': 0, 
                    'current_price': price, 
                    'high_water_mark': price, 
                    'low_water_mark': price,
                    'stop_distance': 0
                }
            else:
                self.positions[symbol]['current_price'] = price
                
                # Update Water Marks for ALL positions (Required for MAE/MFE calculation)
                # Initialize if missing (Migration safety)
                if 'low_water_mark' not in self.positions[symbol] or self.positions[symbol]['low_water_mark'] == 0:
                     self.positions[symbol]['low_water_mark'] = price
                if 'high_water_mark' not in self.positions[symbol]:
                     self.positions[symbol]['high_water_mark'] = price
                     
                # Track Gloabl HWM/LWM during trade
                if price > self.positions[symbol]['high_water_mark']:
                    self.positions[symbol]['high_water_mark'] = price
                if price < self.positions[symbol]['low_water_mark']:
                    self.positions[symbol]['low_water_mark'] = price
            
            
            # DB Snapshot prep (Copy inside lock)
            if symbol in self.positions:
                pos = self.positions[symbol].copy()
                should_update_db = True
            else:
                should_update_db = False
        finally:
            self.guard.release()

    def _update_kelly_stats(self, pnl: float):
        """
        Phase 14: Updates rolling statistics for Dynamic Kelly computation.
        Uses a rolling window of recent trades.
        """
        is_win = pnl > 0
        self.kelly_trades_history.append({'pnl': pnl, 'is_win': is_win})
        
        # Enforce rolling window
        if len(self.kelly_trades_history) > self.KELLY_WINDOW:
            self.kelly_trades_history.pop(0)
            
        # Recompute stats
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        loss_count = 0
        
        for t in self.kelly_trades_history:
            if t['is_win']:
                wins += 1
                gross_profit += t['pnl']
            else:
                loss_count += 1
                gross_loss += abs(t['pnl'])
                
        total_trades = len(self.kelly_trades_history)
        if total_trades > 0:
            self.kelly_winrate = wins / total_trades
            
        avg_win = (gross_profit / wins) if wins > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 1.0 # Prevent div by zero
        
        if avg_loss > 0:
            self.kelly_payoff_ratio = avg_win / avg_loss
        else:
            self.kelly_payoff_ratio = avg_win # If no losses, payoff is basically infinite, but cap at average win
            
    def get_kelly_metrics(self) -> tuple[float, float]:
        """Returns (winrate, payoff_ratio) for Kelly computation"""
        return self.kelly_winrate, self.kelly_payoff_ratio

        # OPTIMIZATION: Throttle disk writes (Debounce)
        now = datetime.now()
        if not hasattr(self, '_last_save_time'): self._last_save_time = datetime.min
        
        if self.auto_save and (now - self._last_save_time).total_seconds() > 1.0:
            self._refresh_equity_cache() # Phase 5: Ensure cache is fresh for Dashboard/Risk
            self.save_status() # Now uses executor
            self._last_save_time = now
        else:
            # Still update cache if price moves significantly (e.g. > 0.05%)
            # to maintain Risk Manager accuracy without too many calculations
            self._refresh_equity_cache()
            
        # Update DB (Snapshot for crash recovery)
        if should_update_db:
            # Calculate Unrealized PnL for DB
            qty = pos['quantity']
            avg = pos['avg_price']
            pnl = (price - avg) * qty if qty != 0 else 0
            
            # Async DB Update
            self.io_executor.submit(self.db.update_position, symbol, qty, avg, price, pnl)

    @trace_execution
    def update_signal(self, event):
        if event.type == EventType.SIGNAL:
            # Phase 9: Capture Entry Metadata (Features, LogProbs, etc.) for PPO
            if event.metadata:
                self.guard.acquire()
                try:
                    # Create entry if needed (Pre-fill before actual order/fill)
                    if event.symbol not in self.positions:
                        self.positions[event.symbol] = {
                            'quantity': 0, 'avg_price': 0, 'current_price': 0,
                            'high_water_mark': 0, 'low_water_mark': 0,
                            'entry_metadata': None
                        }
                    
                    # Store full metadata
                    self.positions[event.symbol]['entry_metadata'] = event.metadata
                finally:
                    self.guard.release()
                
                # SOPHIA-INTELLIGENCE: Store trade intent for Post-Mortem
                sophia_data = event.metadata.get('sophia')
                if sophia_data and hasattr(event, 'trade_id') and event.trade_id:
                    try:
                        self.sophia_post_mortem.store_intent(
                            trade_id=event.trade_id,
                            symbol=event.symbol,
                            direction=event.signal_type.name if hasattr(event.signal_type, 'name') else str(event.signal_type),
                            sophia_report=sophia_data,
                            narrative=event.metadata.get('sophia_narrative', ''),
                            trigger_price=getattr(event, 'current_price', 0.0) or 0.0,
                        )
                    except Exception as e:
                        logger.debug(f"[SOPHIA] Intent store skipped: {e}")

    def update_fill(self, event) -> Optional[Tuple[float, TradeOutcome]]:
        """Atomically update portfolio state. Returns (realized PnL, TradeOutcome) if closed."""
        if event.type == EventType.FILL:
            pnl_realized = None
            outcome_obj = None # Neural Fortress Object
            # Update Cash and Positions
            fill_cost = event.fill_cost # Total notional value (price * quantity)
            fill_price = event.fill_cost / event.quantity if event.quantity > 0 else 0
            
            # Calculate Margin Impact (Futures Only)
            margin_impact = 0.0
            if Config.BINANCE_USE_FUTURES:
                leverage = Config.BINANCE_LEVERAGE
                margin_impact = fill_cost / leverage
            
            # EXACT FEE LOGIC (Phase 17.4 Audit)
            # Use actual commission from FillEvent if available, else estimate
            fee_rate = 0.0006 if Config.BINANCE_USE_FUTURES else 0.001
            
            if event.commission is not None:
                estimated_fee = event.commission
            else:
                # Fallback to estimate
                estimated_fee = fill_cost * fee_rate
            
            
            # BEGIN ATOMIC UPDATE
            self.guard.acquire()
            try:
                # Capture Pre-State for Accounting Audit
                pre_balance = Decimal(str(self.current_cash))
                
                # Deduct fee from Cash immediately (Atomic & Single Deduction)
                self.current_cash -= estimated_fee
                self.total_fees_paid += estimated_fee  # CRITERIO-AXIOMA: explicit fee tracking
                
                logger.info(f"  💸 Fee Paid: ${estimated_fee:.4f} ({fee_rate*100}%)")
                
                if event.direction == OrderSide.BUY:
                    # BUY can be: Close SHORT or Open LONG
                    pos = self.positions.get(event.symbol, {'quantity': 0, 'avg_price': 0})
                    
                    if pos['quantity'] < 0:
                        # === CLOSING SHORT (and potentially FLIPPING) ===
                        short_qty = abs(pos['quantity'])
                        closed_qty = min(short_qty, event.quantity)
                        new_long_qty = max(0, event.quantity - closed_qty)
                        
                        # 1. Calculate PnL for the closed portion
                        entry_price = pos['avg_price']
                        exit_price = fill_price
                        pnl = (entry_price - exit_price) * closed_qty
                        
                        self.realized_pnl += pnl
                        self.current_cash += pnl
                        
                        # Neural Fortress: Trade Outcome Calculation
                        try:
                            lwm = pos.get('low_water_mark', entry_price)
                            hwm = pos.get('high_water_mark', entry_price)
                            entry_time = pos.get('entry_time', datetime.now(timezone.utc))
                            duration = (datetime.now(timezone.utc) - entry_time).total_seconds()
                            
                            # SHORT: MAE is High - Entry (Negative move), MFE is Entry - Low
                            mae = max(0.0, hwm - entry_price)
                            mfe = max(0.0, entry_price - lwm)
                            
                            outcome_obj = TradeOutcome(
                                entry_price=entry_price,
                                exit_price=exit_price,
                                direction=-1,
                                leverage=Config.BINANCE_LEVERAGE if Config.BINANCE_USE_FUTURES else 1.0,
                                max_adverse_excursion=mae,
                                max_favorable_excursion=mfe,
                                duration_seconds=duration,
                                latency_ms=0.0, # Filled by Engine/LatencyMonitor later if needed
                                entry_features=pos.get('entry_metadata', {}).get('features'),
                                metadata=pos.get('entry_metadata')
                            )
                        except Exception as e:
                            logger.error(f"Failed to create TradeOutcome: {e}")

                        # 2. Release Margin for closed portion
                        if Config.BINANCE_USE_FUTURES:
                            # Proportional margin release
                            closed_margin = (closed_qty * fill_price) / Config.BINANCE_LEVERAGE
                            self.used_margin = max(0, self.used_margin - closed_margin)
                        
                        # 3. Update Performance
                        strat_id = getattr(event, 'strategy_id', None) or pos.get('opener_strategy_id', 'Unknown')
                        self._update_strategy_performance(strat_id, pnl)
                        self._update_kelly_stats(pnl) # Phase 14: Dynamic Kelly tracking                        
                        logger.info(f"📈 SHORT Closed: {event.symbol} PnL=${pnl:.2f} (Qty: {closed_qty})")
                        
                        # 4. Handle FLIP (Opening NEW LONG leg)
                        if new_long_qty > 0:
                            pos['quantity'] = new_long_qty
                            pos['avg_price'] = fill_price
                            pos['high_water_mark'] = fill_price
                            pos['low_water_mark'] = fill_price # Init LWM
                            pos['entry_time'] = datetime.now(timezone.utc) # Init Entry Time
                            
                            pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                            pos['sl_pct'] = getattr(event, 'sl_pct', None)
                            pos['tp_pct'] = getattr(event, 'tp_pct', None)
                            
                            # Add margin for the NEW leg
                            if Config.BINANCE_USE_FUTURES:
                                new_margin = (new_long_qty * fill_price) / Config.BINANCE_LEVERAGE
                                self.used_margin += new_margin
                                self.pending_cash = max(0, self.pending_cash - new_margin)
                            
                            logger.info(f"🔄 FLIP: SHORT -> LONG {event.symbol} (New Qty: {new_long_qty} @ ${fill_price:.2f})")
                        else:
                            pos['quantity'] = 0
                            pos['avg_price'] = 0
                            self.positions.pop(event.symbol, None)
    
                        # REPORTING
                        self.log_trade_report(event, pnl=pnl, fill_price=exit_price)
                        pnl_realized = pnl
                        
                        # SOPHIA Post-Mortem (SHORT Close)
                        self._sophia_post_mortem_check(event, pnl, duration)
                        
                        # CRITERIO-AXIOMA Accounting Audit
                        post_balance = Decimal(str(self.current_cash))
                        expected_balance = pre_balance - Decimal(str(estimated_fee)) + Decimal(str(pnl))
                        drift = abs(post_balance - expected_balance)
                        self.precision_drift_accumulated += drift
                        
                        if drift > Decimal('1e-8'):
                             logger.warning(f"⚠️ [AXIOMA-LOG] Precision Drift Detected in SHORT Close: Max Deviation {drift:.4e}")
                        
                        self.verify_accounting_equation()
                    
                    else:
                        # === OPENING/ADDING LONG POSITION ===
                        
                        # Update Cash/Margin
                        if Config.BINANCE_USE_FUTURES:
                            self.used_margin += margin_impact
                            # Release pending (it moves to used_margin)
                            self.pending_cash = max(0, self.pending_cash - margin_impact)
                        else:
                            # Spot: Spend Cash
                            self.current_cash -= fill_cost
                            self.pending_cash = max(0, self.pending_cash - fill_cost)
                        
                        # Update Avg Price
                        if event.symbol not in self.positions:
                            self.positions[event.symbol] = {
                                'quantity': 0, 
                                'avg_price': 0, 
                                'current_price': 0, 
                                'high_water_mark': 0, 
                                'low_water_mark': 0,
                                'stop_distance': 0,
                                'sl_pct': getattr(event, 'sl_pct', None),
                                'tp_pct': getattr(event, 'tp_pct', None),
                                'opener_strategy_id': getattr(event, 'strategy_id', None),
                                'entry_time': datetime.now(timezone.utc)
                            }
                        
                        pos = self.positions[event.symbol]
                        if pos['quantity'] == 0:
                             pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                             pos['entry_time'] = datetime.now(timezone.utc) # Reset time on fresh open
                        
                        total_cost = (pos['quantity'] * pos['avg_price']) + fill_cost
                        total_qty = pos['quantity'] + event.quantity
                        
                        if total_qty > 0:
                            pos['avg_price'] = total_cost / total_qty
                        pos['quantity'] = total_qty
                        
                        pos['high_water_mark'] = max(pos.get('high_water_mark', 0), fill_price)
                        pos['low_water_mark'] = min(pos.get('low_water_mark', 999999), fill_price) # Track LWM too
                        
                        # REPORTING (Entry)
                        self.log_trade_report(event, pnl=None, fill_price=fill_price)
                        
                        # CRITERIO-AXIOMA Accounting Audit (Fee deduction only)
                        post_balance = Decimal(str(self.current_cash))
                        expected_balance = pre_balance - Decimal(str(estimated_fee))
                        if Config.BINANCE_USE_FUTURES is False:
                             from_cash_spent = Decimal(str(fill_cost))
                             expected_balance -= from_cash_spent
                        drift = abs(post_balance - expected_balance)
                        self.precision_drift_accumulated += drift
                    
                elif event.direction == OrderSide.SELL:
                    # SELL can be: Close LONG or Open SHORT
                    pos = self.positions.get(event.symbol, {'quantity': 0, 'avg_price': 0})
                    
                    if pos['quantity'] > 0:
                        # === CLOSING LONG (and potentially FLIPPING) ===
                        long_qty = pos['quantity']
                        closed_qty = min(long_qty, event.quantity)
                        new_short_qty = max(0, event.quantity - closed_qty)
                        
                        # 1. Calculate PnL for closed portion
                        pnl = (fill_price - pos['avg_price']) * closed_qty
                        self.realized_pnl += pnl
                        pnl_realized = pnl
                        
                        # Neural Fortress: Trade Outcome Calculation (LONG Close)
                        try:
                            lwm = pos.get('low_water_mark', pos['avg_price'])
                            hwm = pos.get('high_water_mark', pos['avg_price'])
                            entry_time = pos.get('entry_time', datetime.now(timezone.utc))
                            duration = (datetime.now(timezone.utc) - entry_time).total_seconds()
                            
                            
                            # LONG: MAE is Entry - Low, MFE is High - Entry
                            mae = max(0.0, pos['avg_price'] - lwm)
                            mfe = max(0.0, hwm - pos['avg_price'])
                            
                            outcome_obj = TradeOutcome(
                                entry_price=pos['avg_price'],
                                exit_price=fill_price,
                                direction=1,
                                leverage=Config.BINANCE_LEVERAGE if Config.BINANCE_USE_FUTURES else 1.0,
                                max_adverse_excursion=mae,
                                max_favorable_excursion=mfe,
                                duration_seconds=duration,
                                latency_ms=0.0,
                                entry_features=pos.get('entry_metadata', {}).get('features'),
                                metadata=pos.get('entry_metadata')
                            )
                        except Exception as e:
                            logger.error(f"Failed to create TradeOutcome (LONG): {e}")
                        
                        # 2. Update Cash/Margin
                        if Config.BINANCE_USE_FUTURES:
                            self.current_cash += pnl
                            closed_margin = (closed_qty * fill_price) / Config.BINANCE_LEVERAGE
                            self.used_margin = max(0, self.used_margin - closed_margin)
                        else:
                            self.current_cash += (closed_qty * fill_price)
                        
                        # 3. Update Performance
                        strat_id = getattr(event, 'strategy_id', None) or pos.get('opener_strategy_id', 'Unknown')
                        self._update_strategy_performance(strat_id, pnl)
                        self._update_kelly_stats(pnl) # Phase 14: Dynamic Kelly tracking
                        
                        # 4. Handle FLIP (Opening NEW SHORT leg)
                        if new_short_qty > 0:
                            pos['quantity'] = -new_short_qty
                            pos['avg_price'] = fill_price
                            pos['low_water_mark'] = fill_price
                            pos['high_water_mark'] = fill_price # Init HWM
                            pos['entry_time'] = datetime.now(timezone.utc)
                            
                            pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                            pos['sl_pct'] = getattr(event, 'sl_pct', None)
                            pos['tp_pct'] = getattr(event, 'tp_pct', None)
                            
                            if Config.BINANCE_USE_FUTURES:
                                new_margin = (new_short_qty * fill_price) / Config.BINANCE_LEVERAGE
                                self.used_margin += new_margin
                                self.pending_cash = max(0, self.pending_cash - new_margin)
                            
                            logger.info(f"🔄 FLIP: LONG -> SHORT {event.symbol} (New Qty: {new_short_qty} @ ${fill_price:.2f})")
                        else:
                            pos['quantity'] = 0
                            pos['avg_price'] = 0
                            self.positions.pop(event.symbol, None)
                            
                        # REPORTING
                        self.log_trade_report(event, pnl=pnl, fill_price=fill_price)
                        
                        # SOPHIA Post-Mortem (LONG Close)
                        self._sophia_post_mortem_check(event, pnl, duration)
                        
                        # CRITERIO-AXIOMA Accounting Audit
                        post_balance = Decimal(str(self.current_cash))
                        expected_balance = pre_balance - Decimal(str(estimated_fee)) + Decimal(str(pnl))
                        if Config.BINANCE_USE_FUTURES is False:
                             sold_cash_returned = Decimal(str(closed_qty * fill_price))
                             expected_balance = pre_balance - Decimal(str(estimated_fee)) + sold_cash_returned

                        drift = abs(post_balance - expected_balance)
                        self.precision_drift_accumulated += drift
                        
                        if drift > Decimal('1e-8'):
                             logger.warning(f"⚠️ [AXIOMA-LOG] Precision Drift Detected in LONG Close: Max Deviation {drift:.4e}")
                        
                        # Trigger Check
                        if self.precision_drift_accumulated > (Decimal(str(self.initial_capital)) * Decimal('0.00001')):
                             logger.critical(f"🛑 [AXIOMA-LOG] Accumulated Drift Exceeds Tolerance: {self.precision_drift_accumulated:.6e}. Force Sync required.")
                        
                        self.verify_accounting_equation()
                    
                
                # Snapshot for internal use (already inside lock)
                pos_final = self.positions.get(event.symbol, {'quantity': 0, 'avg_price': 0}).copy()
                
            finally:
                self.guard.release()
            
            # Log Trade
            self.log_to_csv({
                'datetime': datetime.now(timezone.utc),
                'symbol': event.symbol,
                'type': EventType.FILL,
                'direction': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'fill_cost': event.quantity * fill_price,
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                'details': f"Exchange: {event.exchange} | Margin: {margin_impact:.2f}"
            })

            # ATOMIC DB UPDATE (Rule 5.2) - Done outside spinlock to avoid blocking loop
            trade_payload = {
                'symbol': event.symbol,
                'side': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'timestamp': datetime.now(timezone.utc),
                'order_type': OrderType.MARKET,
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                'pnl': 0.0, 
                'commission': estimated_fee
            }
            
            position_payload = {
                'symbol': event.symbol,
                'quantity': pos_final['quantity'],
                'entry_price': pos_final['avg_price'],
                'current_price': fill_price,
                'pnl': 0.0
            }
            
            self.db.log_fill_event_atomic(trade_payload, position_payload)
            
            if self.auto_save:
                self.save_status()
                
            if pnl_realized is not None:
                return (pnl_realized, outcome_obj)
            return None

    def verify_accounting_equation(self):
        """
        CRITERIO-AXIOMA Protocol: Ley de Conservación
        Verifica que el Dinero no aparece ni desaparece de la nada.
        Equation: Current_Cash + Used_Margin == Initial_Capital + Realized_PnL - Total_Fees
        (Unrealized PnL is excluded as it fluctuates tick by tick without settling)
        """
        # Calculate theoretical settled balance
        theoretical_settled = self.initial_capital + self.realized_pnl - self.total_fees_paid
        actual_settled = self.current_cash + self.used_margin
        
        try:
            from decimal import Decimal
            d_theoretical = Decimal(str(theoretical_settled))
            d_actual = Decimal(str(actual_settled))
            delta = abs(d_theoretical - d_actual)
            
            if delta > PrecisionAuditor.STRICT_EPSILON:
                logger.error(f"🚨 [AXIOMA-FATAL] CORRUPCIÓN CONTABLE DETECTADA!")
                logger.error(f"   Teórico:  ${theoretical_settled:.8f}")
                logger.error(f"   Real:     ${actual_settled:.8f}")
                logger.error(f"   Delta:    ${delta.normalize():f}")
                logger.error(f"   Initial={self.initial_capital}, PnL={self.realized_pnl}, Fees={self.total_fees_paid}")
                
                # Soft-kill the engine by poisoning PnL
                logger.error("☠️ SISTEMA COMPROMETIDO. CÁLCULOS INVÁLIDOS.")
                self.realized_pnl = float('nan')
        except Exception as e:
            logger.error(f"⚠️ [AXIOMA] Falló la auditoría contable: {e}")

    def _sophia_post_mortem_check(self, event, pnl: float, duration_seconds: float = 0.0):
        """
        SOPHIA-INTELLIGENCE + NÉMESIS-RETROSPECCIÓN:
        1. Compute SOPHIA Brier Score (basic post-mortem)
        2. Run NÉMESIS full autopsy (deep diagnosis)
        Called from update_fill() when a position is closed (SHORT or LONG).
        """
        try:
            trade_id = getattr(event, 'trade_id', None)
            if not trade_id:
                # Try to find by symbol match
                for tid in list(self.sophia_post_mortem.pending_intents.keys()):
                    intent = self.sophia_post_mortem.pending_intents[tid]
                    if intent.symbol == event.symbol:
                        trade_id = tid
                        break
            
            if trade_id:
                # Retrieve intent BEFORE compute_post_mortem pops it
                intent = self.sophia_post_mortem.pending_intents.get(trade_id)
                
                result = self.sophia_post_mortem.compute_post_mortem(
                    trade_id=trade_id,
                    actual_pnl=pnl,
                    duration_seconds=duration_seconds,
                )
                if result:
                    # Update SOPHIA calibrator if available
                    won = pnl > 0
                    # Log calibration status periodically
                    if self.sophia_post_mortem.total_trades % 10 == 0:
                        cal_log = self.sophia_post_mortem.get_summary_log()
                        logger.info(f"   📊 {cal_log}")
                    
                    # ── NÉMESIS-RETROSPECCIÓN: Full Autopsy ──
                    if intent:
                        try:
                            fill_price = getattr(event, 'fill_price', 0.0) or 0.0
                            sophia_data = intent.sophia_report or {}
                            
                            nemesis_report = self.nemesis_engine.full_autopsy(
                                trade_id=trade_id,
                                symbol=event.symbol,
                                direction=intent.direction,
                                predicted_prob=intent.win_probability,
                                predicted_exit_mins=intent.expected_exit_mins,
                                predicted_tp_mins=sophia_data.get('time_to_tp_mins', 10.0),
                                predicted_sl_mins=sophia_data.get('time_to_sl_mins', 5.0),
                                actual_pnl=pnl,
                                actual_duration_mins=duration_seconds / 60.0,
                                brier_score=result.brier_score,
                                sophia_report=sophia_data,
                                top_features=intent.top_features,
                                trigger_price=intent.trigger_price,
                                fill_price=fill_price,
                            )
                            
                            # Apply overconfidence penalty to next signals
                            if nemesis_report.overconfidence_active:
                                logger.info(
                                    f"   ⚠️ [NÉMESIS] Confidence penalty active: "
                                    f"{nemesis_report.overconfidence_penalty_factor:.2f}x"
                                )
                            
                            # Log NÉMESIS health every 10 trades
                            if self.sophia_post_mortem.total_trades % 10 == 0:
                                health = self.nemesis_engine.get_calibration_health()
                                logger.info(f"   ⚔️ [NÉMESIS] Health: {health}")
                        except Exception as e:
                            logger.debug(f"[NÉMESIS] Autopsy skipped: {e}")
        except Exception as e:
            logger.debug(f"[SOPHIA] Post-mortem skipped: {e}")

    def close(self):
        """
        Graceful shutdown for portfolio resources.
        """
        if hasattr(self, 'db') and self.db:
            self.db.close()
        logger.info("✅ Portfolio: Shutdown complete.")

    def save_status(self):
        """
        Save current portfolio state.
        Uses ThreadPoolExecutor to prevent I/O blocking.
        """
        # Snapshot state safely
        self.guard.acquire()
        try:
            cash_snapshot = self.current_cash
            realized_snapshot = self.realized_pnl
            positions_snapshot = self.positions.copy()
        finally:
            self.guard.release()
            
        equity = self.get_total_equity()
        unrealized_pnl = equity - self.initial_capital - realized_snapshot
        
        # Submit to Executor
        self.io_executor.submit(self._do_save_status, cash_snapshot, realized_snapshot, positions_snapshot, equity, unrealized_pnl)

    def _do_save_status(self, cash, realized, positions, equity, unrealized):
        """Worker method for save_status: Executed in background thread"""
        try:
            # Get Session Info (Disabled: Decoupled for tests)
            session_id = None
            
            metrics = {} 
            
            # SPECTACULAR OPTIMIZATION: Heavy Math is now here, off the main Event Loop!
            try:
                # Load trades safely for Expectancy
                trades_path = os.path.join(os.path.dirname(self.status_path), "trades.csv")
                # Use thread-safe read (F43 Fix)
                trades_df = safe_read_csv(trades_path)
                
                if trades_df is not None:
                    
                    # Lazy Import to avoid circular dependencies
                    from utils.analytics import AnalyticsEngine 
                    
                    # Complex generic calculations
                    exp_stats = AnalyticsEngine.calculate_expectancy(trades_df)
                    metrics.update(exp_stats)
                    
                    # Also calculate Sharpe/Sortino if enough data (approximate)
                    if len(trades_df) > 30:
                         # Assume 1m signals for simplicity or use PnL curve
                         pass # Keep it simple for now to avoid crashes
            except Exception as e:
                # Don't fail the save if analytics fail
                logger.debug(f"Async Analytics Calc Skipped: {e}")

            json_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': session_id,
                'total_equity': equity,
                'cash': cash,
                'realized_pnl': realized,
                'unrealized_pnl': unrealized,
                'positions': positions,
                'performance_metrics': metrics, # Now populated!
                'balance': cash, 
                'precision_drift': float(getattr(self, 'precision_drift_accumulated', 0.0)),
                'last_heartbeat': datetime.now(timezone.utc).isoformat()
            }
            
            # 🛡️ PHASE 27: ATOMIC PERSISTENCE (Nadir-Soberano)
            # Replaces legacy handler with AtomicStateManager
            json_path = os.path.join(os.path.dirname(self.status_path), "live_status.json")
            AtomicStateManager.save_json_atomic(json_path, json_data)
        except Exception as e:
            logger.error(f"Async Status Save Failed: {e}")

    def _sync_log_to_csv(self, data):
        """Internal synchronous method for CSV writing"""
        try:
            # Use thread-safe append (F43 Fix)
            safe_append_csv(self.csv_path, data)
        except Exception as e:
            logger.error(f"CSV Log Failed: {e}")


    @trace_execution
    def check_exits(self, data_provider, events_queue):
        """
        LAYER 1: Portfolio-based exit monitoring.
        Checks all open positions and generates EXIT signals based on PnL thresholds.
        This is a safety net that runs regardless of strategy logic.
        
        Thresholds:
        - Stop Loss: -0.3% (cut losses quickly)
        - Take Profit: +0.8% (lock in profits)
        - Trailing Stop: -0.2% from peak (protect profits)
        """
        from core.events import SignalEvent
        
        # Snapshot iteration keys to avoid RuntimeError
        self.guard.acquire()
        try:
            # We must iterate a Copy to allow internal logic to potentially trigger closes (which modify dict)
            snapshot_items = list(self.positions.items())
        finally:
            self.guard.release()

        now_utc = datetime.now(timezone.utc)

        for symbol, position in snapshot_items:
            # Skip if no position
            if position['quantity'] == 0:
                continue
                
            current_price = position.get('current_price', 0)
            entry_price = position.get('avg_price', 0)
            quantity = position['quantity']
            
            # Skip if price not available
            if current_price == 0 or entry_price == 0:
                continue
            
            # Calculate PnL percentage
            if quantity > 0:  # LONG position
                pnl_pct = (current_price - entry_price) / entry_price
                hwm = position.get('high_water_mark', current_price)
                drawdown_from_peak = (current_price - hwm) / hwm
                
                # LONG Exit Conditions
                if pnl_pct < -0.003:  # Stop Loss: -0.3%
                    logger.warning(f"🛑 STOP LOSS triggered for {symbol}: {pnl_pct*100:.2f}%")
                    events_queue.put(SignalEvent("99", symbol, now_utc, SignalType.EXIT, strength=1.0))
                    
                elif pnl_pct > 0.008:  # Take Profit: +0.8%
                    logger.info(f"💰 TAKE PROFIT triggered for {symbol}: {pnl_pct*100:.2f}%")
                    events_queue.put(SignalEvent("99", symbol, now_utc, SignalType.EXIT, strength=1.0))
                    
                elif drawdown_from_peak < -0.002 and pnl_pct > 0.002:  # Trailing Stop: -0.2% from peak (only if in profit)
                    logger.info(f"📉 TRAILING STOP triggered for {symbol}: Peak {hwm:.4f}, Now {current_price:.4f}")
                    events_queue.put(SignalEvent("99", symbol, now_utc, SignalType.EXIT, strength=1.0))
                    
            elif quantity < 0:  # SHORT position
                pnl_pct = (entry_price - current_price) / entry_price
                lwm = position.get('low_water_mark', current_price)
                drawup_from_low = (lwm - current_price) / lwm
                
                # SHORT Exit Conditions
                if pnl_pct < -0.003:  # Stop Loss: -0.3%
                    logger.warning(f"🛑 STOP LOSS triggered for SHORT {symbol}: {pnl_pct*100:.2f}%")
                    events_queue.put(SignalEvent("99", symbol, now_utc, SignalType.EXIT, strength=1.0))
                    
                elif pnl_pct > 0.008:  # Take Profit: +0.8%
                    logger.info(f"💰 TAKE PROFIT triggered for SHORT {symbol}: {pnl_pct*100:.2f}%")
                    events_queue.put(SignalEvent("99", symbol, now_utc, SignalType.EXIT, strength=1.0))
                    
                elif drawup_from_low < -0.002 and pnl_pct > 0.002:  # Trailing Stop
                    logger.info(f"📈 TRAILING STOP triggered for SHORT {symbol}: Low {lwm:.4f}, Now {current_price:.4f}")
                    events_queue.put(SignalEvent("99", symbol, now_utc, SignalType.EXIT, strength=1.0))

    @trace_execution
    def get_smart_kelly_sizing(self, symbol: str, strategy_id: str) -> float:
        """
        🚀 DYNAMIC CAPITAL ALLOCATION (Smart Kelly - Phase 13)
        Uses real historical Payoff Ratio (Avg Win / Avg Loss) for precise sizing.
        """
        perf = self.strategy_performance.get(strategy_id, {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0})
        total_trades = perf['trades']
        
        # Bootstrap default if insufficient data
        if total_trades < 10: return 0.02
        
        wins = perf['wins']
        losses = perf['losses']
        
        if wins == 0: return 0.005 # Minimal size if no wins yet
        if losses == 0: return 0.05 # Max cap if no losses yet (start strong)
        
        win_rate = wins / total_trades
        loss_rate = 1.0 - win_rate
        
        # Calculate Real Payoff Ratio (b)
        total_win_amt = perf.get('total_win_pnl', 0.0)
        total_loss_amt = perf.get('total_loss_pnl', 0.0)
        
        avg_win = total_win_amt / wins
        avg_loss = total_loss_amt / losses if losses > 0 else 1.0
        
        if avg_loss == 0: b = 2.0 # Safety default
        else: b = avg_win / avg_loss
        
        # Kelly Formula: f = p - q/b
        if b <= 0: return 0.005
        
        kelly_f = win_rate - (loss_rate / b)
        
        # Fractional Kelly (Safety Factor 0.5)
        # Cap at 5% risk per trade for HFT
        final_size = max(0.005, min(0.05, kelly_f * 0.5))
        
        return final_size

    def log_trade_report(self, event, pnl=None, fill_price=0):
        """Prints a real-time report of the trade execution, Win Rate, and Balance."""
        try:
            # 1. Global Performance Stats
            total_wins = sum(d['wins'] for d in self.strategy_performance.values())
            total_losses = sum(d['losses'] for d in self.strategy_performance.values())
            total_trades = sum(d['trades'] for d in self.strategy_performance.values())
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            
            # 2. Balance Stats
            equity = self.get_total_equity()
            balance_delta = equity - self.initial_capital
            balance_pct = (balance_delta / self.initial_capital) * 100
            
            direction_icon = "🟢 BUY" if event.direction == OrderSide.BUY else "🔴 SELL"
            pnl_str = f"+${pnl:.2f}" if pnl and pnl > 0 else (f"-${abs(pnl):.2f}" if pnl else "N/A")
            pnl_color = "🟢" if pnl and pnl > 0 else ("🔴" if pnl and pnl < 0 else "⚪")
            
            print("\n📢 ================= [ TRADE EXECUTION ] =================", flush=True)
            print(f"   {direction_icon} {event.symbol} @ ${fill_price:.4f} (Qty: {event.quantity})", flush=True)
            if pnl is not None:
                print(f"   💰 PnL Realized: {pnl_color} {pnl_str}", flush=True)
            
            print(f"   🏆 Win Rate:     {win_rate:.1f}% ({total_wins} Wins / {total_losses} Losses)", flush=True)
            print(f"   💵 Net Equity:   ${equity:.2f} ({'+' if balance_delta >=0 else ''}{balance_pct:.2f}%)", flush=True)
            print("========================================================\n", flush=True)
            
            # --- NOTIFICACIÓN EXTERNA (Phase 4) ---
            Notifier.notify_trade(
                symbol=event.symbol,
                direction=event.direction,
                price=fill_price,
                qty=event.quantity,
                pnl=pnl,
                winrate=win_rate
            )
        except Exception as e:
            logger.error(f"⚠️ Report Error: {e}")

    def log_to_csv(self, data):
        self.io_executor.submit(self._sync_log_to_csv, data)

    def _update_strategy_performance(self, strategy_id: str, pnl: float):
        """Helper to update strategy performance stats including PnL sums for Kelly."""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                'trades': 0, 'wins': 0, 'losses': 0, 
                'pnl': 0.0, 'win_rate': 0.0,
                'total_win_pnl': 0.0, 'total_loss_pnl': 0.0
            }
            
        stats = self.strategy_performance[strategy_id]
        stats['trades'] += 1
        stats['pnl'] += pnl
        
        if pnl > 0:
            stats['wins'] += 1
            stats['total_win_pnl'] = stats.get('total_win_pnl', 0.0) + pnl
        elif pnl < 0:
            stats['losses'] += 1
            stats['total_loss_pnl'] = stats.get('total_loss_pnl', 0.0) + abs(pnl)
            
        if stats['trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['trades']

    def get_strategy_metrics(self, strategy_id: str) -> Dict[str, float]:
        """
        Phase 14: Return real-time performance metrics for a specific strategy.
        Used by RiskManager for Kelly Criterion sizing.
        """
        strat_data = self.strategy_performance.get(strategy_id, {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0})
        
        wins = strat_data['wins']
        losses = strat_data['losses']
        total = wins + losses # Using completed trades only
        
        win_rate = (wins / total) if total > 0 else 0.5 # Default 50% assumption
        
        # Calculate Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
        # We need avg win/loss. 
        # Ideally strategy_performance should track total_win_amt and total_loss_amt separately.
        # For now, we approximation or we need to update strategy_performance structure.
        # Let's keep it simple for now and just return Win Rate and Profit Factor proxy (PnL)
        
        return {
            'win_rate': win_rate,
            'total_pnl': strat_data['pnl'],
            'total_trades': strat_data['trades']
        }
