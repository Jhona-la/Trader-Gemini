
import pandas as pd
import os
import json
from datetime import datetime, timezone
from config import Config  # Import Config for Futures/Spot detection
from core.enums import EventType, SignalType, OrderSide, OrderType, TradeDirection, TradeStatus
from core.data_handler import get_data_handler
from utils.analytics import AnalyticsEngine
from utils.logger import logger
from utils.transparent_logger import TransparentLogger
from utils.debug_tracer import trace_execution
from utils.position_cleaner import position_cleaner
from utils.notifier import Notifier
from utils.session_manager import get_session_manager
from utils.data_manager import DatabaseHandler

from typing import Dict, Optional, List, Any, Union

class Portfolio:
    def __init__(self, initial_capital: float = 10000.0, 
                 csv_path: str = "dashboard/data/trades.csv", 
                 status_path: str = "dashboard/data/status.csv", 
                 auto_save: bool = True):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.pending_cash = 0.0  # Cash reserved for pending orders
        self.used_margin = 0.0   # Margin locked in Futures positions
        
        self.positions = {} # Symbol -> {'quantity': 0, 'avg_price': 0, 'current_price': 0}
        self.realized_pnl = 0.0
        
        # STRATEGY ATTRIBUTION: Track PnL per strategy
        # Format: {strategy_id: {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0}}
        self.strategy_performance = {}
        
        self.csv_path = csv_path
        self.status_path = status_path
        self.auto_save = auto_save
        
        # Thread safety for cash operations
        import threading
        self._cash_lock = threading.RLock()
        
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
        
    def update_math_stats(self, stats: Dict[str, Any]) -> None:
        """Update live mathematical statistics from strategies."""
        if not stats: return
        self.math_stats.update(stats)
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
            with open(state_path, 'r') as f:
                data = json.load(f)
                
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
    
    @trace_execution
    def get_available_cash(self):
        """Return cash available for trading (Total Cash - Margin Used - Pending)."""
        with self._cash_lock:
            # In Futures: Cash is collateral. Available = Cash - Margin - Pending
            # In Spot: Cash is currency. Available = Cash - Pending (Margin is N/A)
            if Config.BINANCE_USE_FUTURES:
                return self.current_cash - self.used_margin - self.pending_cash
            else:
                return self.current_cash - self.pending_cash

    @property
    def unrealized_pnl(self):
        """
        Calculate total unrealized PnL from all open positions.
        """
        pnl = 0.0
        for symbol, pos in self.positions.items():
            qty = pos['quantity']
            if qty != 0:
                avg_price = pos['avg_price']
                current_price = pos.get('current_price', avg_price)
                # PnL = (Current - Avg) * Qty
                # Works for both LONG (Qty > 0) and SHORT (Qty < 0)
                pnl += (current_price - avg_price) * qty
        return pnl

    def get_total_equity(self):
        """
        Return Total Equity = Cash + Unrealized PnL of all open positions.
        Used by RiskManager for position sizing.
        """
        equity = self.current_cash
        
        # Add Unrealized PnL from open positions
        for symbol, pos in self.positions.items():
            qty = pos['quantity']
            if qty != 0:
                avg_price = pos['avg_price']
                current_price = pos.get('current_price')
                
                # Safety check: Ensure both prices are valid numbers
                if current_price is None:
                    current_price = 0.0
                if avg_price is None:
                    avg_price = 0.0
                    
                # If we have no price data, assume 0 PnL change (use avg as current)
                if current_price == 0 and avg_price > 0:
                    current_price = avg_price
                
                # PnL = (Current - Avg) * Qty
                # Works for both LONG (Qty > 0) and SHORT (Qty < 0)
                unrealized_pnl = (current_price - avg_price) * qty
                equity += unrealized_pnl
                
        return equity
    
    @trace_execution
    def reserve_cash(self, amount):
        """Reserve cash for a pending order. Returns True if successful."""
        with self._cash_lock:
            available = self.get_available_cash() # Use internal method to respect margin
            # Note: We don't subtract pending here because get_available_cash already does
            # Wait, get_available_cash calculates *current* available.
            # We need to check if (Available >= amount)
            
            # Recalculate locally to be safe with lock
            if Config.BINANCE_USE_FUTURES:
                avail = self.current_cash - self.used_margin - self.pending_cash
            else:
                avail = self.current_cash - self.pending_cash
                
            if avail >= amount:
                self.pending_cash += amount
                return True
            return False
    
    def release_cash(self, amount):
        """Release reserved cash (order failed/canceled)."""
        with self._cash_lock:
            self.pending_cash = max(0, self.pending_cash - amount)

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
            
            # Update High Water Mark for LONG positions
            if self.positions[symbol]['quantity'] > 0:
                if price > self.positions[symbol].get('high_water_mark', 0):
                    self.positions[symbol]['high_water_mark'] = price
            
            # Update Low Water Mark for SHORT positions
            elif self.positions[symbol]['quantity'] < 0:
                lwm = self.positions[symbol].get('low_water_mark', price)
                if price < lwm:
                    self.positions[symbol]['low_water_mark'] = price
        
        # OPTIMIZATION: Throttle disk writes (Debounce)
        # Only save if > 1s passed, UNLESS it's a critical update (not passed here yet, so assuming all price updates are non-critical)
        now = datetime.now()
        if not hasattr(self, '_last_save_time'): self._last_save_time = datetime.min
        
        if self.auto_save and (now - self._last_save_time).total_seconds() > 1.0:
            self.save_status()
            self._last_save_time = now
            
        # Update DB (Snapshot for crash recovery)
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Calculate Unrealized PnL for DB
            qty = pos['quantity']
            avg = pos['avg_price']
            pnl = (price - avg) * qty if qty != 0 else 0
            
            self.db.update_position(
                symbol=symbol,
                quantity=qty,
                entry_price=avg,
                current_price=price,
                pnl=pnl
            )

    @trace_execution
    def update_signal(self, event):
        if event.type == EventType.SIGNAL:
            pass

    def update_fill(self, event):
        if event.type == EventType.FILL:
            # Update Cash and Positions
            fill_cost = event.fill_cost # Total notional value (price * quantity)
            fill_price = event.fill_cost / event.quantity if event.quantity > 0 else 0
            
            # Calculate Margin Impact (Futures Only)
            margin_impact = 0.0
            if Config.BINANCE_USE_FUTURES:
                leverage = Config.BINANCE_LEVERAGE
                margin_impact = fill_cost / leverage
            
            # ESTIMATE FEES (Conservative Taker Fee)
            # Binance Futures Taker: ~0.05%
            # Binance Spot Taker: ~0.1%
            # We use 0.06% for Futures and 0.1% for Spot to be safe
            fee_rate = 0.0006 if Config.BINANCE_USE_FUTURES else 0.001
            estimated_fee = fill_cost * fee_rate
            
            # Deduct fee from Cash immediately (Atomic & Single Deduction)
            with self._cash_lock:
                self.current_cash -= estimated_fee
            
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
                    with self._cash_lock:
                         self.current_cash += pnl
                    
                    # 2. Release Margin for closed portion
                    if Config.BINANCE_USE_FUTURES:
                        # Proportional margin release
                        closed_margin = (closed_qty * fill_price) / Config.BINANCE_LEVERAGE
                        self.used_margin = max(0, self.used_margin - closed_margin)
                    
                    # 3. Update Performance
                    strat_id = getattr(event, 'strategy_id', None) or pos.get('opener_strategy_id', 'Unknown')
                    if strat_id not in self.strategy_performance:
                        self.strategy_performance[strat_id] = {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0}
                    self.strategy_performance[strat_id]['pnl'] += pnl
                    self.strategy_performance[strat_id]['trades'] += 1
                    if pnl > 0: self.strategy_performance[strat_id]['wins'] += 1
                    elif pnl < 0: self.strategy_performance[strat_id]['losses'] += 1
                    
                    logger.info(f"📈 SHORT Closed: {event.symbol} PnL=${pnl:.2f} (Qty: {closed_qty})")
                    
                    # 4. Handle FLIP (Opening NEW LONG leg)
                    if new_long_qty > 0:
                        pos['quantity'] = new_long_qty
                        pos['avg_price'] = fill_price
                        pos['high_water_mark'] = fill_price
                        pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                        
                        # Add margin for the NEW leg
                        if Config.BINANCE_USE_FUTURES:
                            new_margin = (new_long_qty * fill_price) / Config.BINANCE_LEVERAGE
                            with self._cash_lock:
                                self.used_margin += new_margin
                                self.pending_cash = max(0, self.pending_cash - new_margin)
                        
                        logger.info(f"🔄 FLIP: SHORT -> LONG {event.symbol} (New Qty: {new_long_qty} @ ${fill_price:.2f})")
                    else:
                        pos['quantity'] = 0
                        pos['avg_price'] = 0
                        position_cleaner.clean_position(event.symbol, pos)

                    # REPORTING
                    self.log_trade_report(event, pnl=pnl, fill_price=exit_price)

                    # CRITICAL FIX: Removed incorrect line that was:
                    # self.pending_cash = max(0, self.pending_cash - margin_impact)
                    # Margin is already released from used_margin on line 195 above.
                    # This line was redundant and caused margin accounting errors.
                
                else:
                    # === OPENING/ADDING LONG POSITION ===
                    
                    # Update Cash/Margin
                    with self._cash_lock:
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
                            'opener_strategy_id': getattr(event, 'strategy_id', None)
                        }
                    
                    pos = self.positions[event.symbol]
                    if pos['quantity'] == 0:
                         pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                    
                    total_cost = (pos['quantity'] * pos['avg_price']) + fill_cost
                    total_qty = pos['quantity'] + event.quantity
                    
                    if total_qty > 0:
                        pos['avg_price'] = total_cost / total_qty
                    pos['quantity'] = total_qty
                    
                    pos['high_water_mark'] = max(pos.get('high_water_mark', 0), fill_price)
                    
                    # REPORTING (Entry)
                    self.log_trade_report(event, pnl=None, fill_price=fill_price)
                
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
                    
                    # 2. Update Cash/Margin
                    if Config.BINANCE_USE_FUTURES:
                        with self._cash_lock:
                             self.current_cash += pnl
                        closed_margin = (closed_qty * fill_price) / Config.BINANCE_LEVERAGE
                        self.used_margin = max(0, self.used_margin - closed_margin)
                    else:
                        with self._cash_lock:
                             self.current_cash += (closed_qty * fill_price)
                    
                    # 3. Update Performance
                    strat_id = getattr(event, 'strategy_id', None) or pos.get('opener_strategy_id', 'Unknown')
                    if strat_id not in self.strategy_performance:
                        self.strategy_performance[strat_id] = {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0}
                    self.strategy_performance[strat_id]['pnl'] += pnl
                    self.strategy_performance[strat_id]['trades'] += 1
                    if pnl > 0: self.strategy_performance[strat_id]['wins'] += 1
                    elif pnl < 0: self.strategy_performance[strat_id]['losses'] += 1
                        
                    logger.info(f"💰 LONG Closed: {event.symbol} PnL=${pnl:.2f} (Qty: {closed_qty})")
                    
                    # 4. Handle FLIP (Opening NEW SHORT leg)
                    if new_short_qty > 0:
                        pos['quantity'] = -new_short_qty
                        pos['avg_price'] = fill_price
                        pos['low_water_mark'] = fill_price
                        pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                        
                        if Config.BINANCE_USE_FUTURES:
                            new_margin = (new_short_qty * fill_price) / Config.BINANCE_LEVERAGE
                            with self._cash_lock:
                                self.used_margin += new_margin
                                self.pending_cash = max(0, self.pending_cash - new_margin)
                        
                        logger.info(f"🔄 FLIP: LONG -> SHORT {event.symbol} (New Qty: {new_short_qty} @ ${fill_price:.2f})")
                    else:
                        pos['quantity'] = 0
                        pos['avg_price'] = 0
                        position_cleaner.clean_position(event.symbol, pos)
                        
                    # REPORTING
                    self.log_trade_report(event, pnl=pnl, fill_price=fill_price)
                
                else:
                    # === OPENING SHORT POSITION ===
                    # Initialize position if needed
                    if event.symbol not in self.positions:
                        self.positions[event.symbol] = {
                            'quantity': 0,
                            'avg_price': 0,
                            'current_price': 0,
                            'high_water_mark': 0,
                            'low_water_mark': 0,
                            'stop_distance': 0,
                            'opener_strategy_id': getattr(event, 'strategy_id', None)
                        }
                    
                    pos = self.positions[event.symbol]
                    
                    # Update Cash/Margin
                    with self._cash_lock:
                        if Config.BINANCE_USE_FUTURES:
                            self.used_margin += margin_impact
                            self.pending_cash = max(0, self.pending_cash - margin_impact)
                        # Spot Shorting not supported in this simple model (requires borrowing)
                    
                    pos['quantity'] = -event.quantity
                    pos['avg_price'] = fill_price
                    pos['low_water_mark'] = fill_price
                    pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                    
                    logger.info(f"📉 SHORT Opened: {event.symbol} @ ${fill_price:.4f} (Qty: {event.quantity})")
                    
                    # REPORTING (Entry)
                    self.log_trade_report(event, pnl=None, fill_price=fill_price)
            
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
            
            # Log Trade to DB
            # ATOMIC DB UPDATE (Rule 5.2)
            pos = self.positions[event.symbol]
            
            trade_payload = {
                'symbol': event.symbol,
                'side': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'timestamp': datetime.now(timezone.utc),
                'order_type': OrderType.MARKET,
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                'pnl': 0.0, # Filled PnL (realized) could be passed here if calculated
                'commission': estimated_fee
            }
            
            position_payload = {
                'symbol': event.symbol,
                'quantity': pos['quantity'],
                'entry_price': pos['avg_price'],
                'current_price': fill_price,
                'pnl': 0.0
            }
            
            self.db.log_fill_event_atomic(trade_payload, position_payload)
            
            if self.auto_save:
                self.save_status()

    def close(self):
        """
        Graceful shutdown for portfolio resources.
        """
        logger.info("Portfolio: Closing database connections...")
        if hasattr(self, 'db') and self.db:
            self.db.close()
        logger.info("✅ Portfolio: Shutdown complete.")

    def save_status(self):
        """
        Save current portfolio state using DataHandler (Phase 5).
        Includes pre-calculated analytics for Dashboard efficiency.
        """
        equity = self.get_total_equity()
        unrealized_pnl = equity - self.initial_capital - self.realized_pnl
        
        # Get Session Info
        session_mgr = get_session_manager()
        session_id = session_mgr.get_session_id() if session_mgr else None
        
        # 0. Pre-calculate Analytics (Phase 5 Efficiency)
        # Load recent history strictly for these calculations
        metrics = {}
        try:
            # We need history to calc sharpe, but Expectancy uses trades
            # For now, let's load minimal history if possible or calculate from what we have
            # Ideally this should be optimized, but using AnalyticsEngine directly is robust
            if hasattr(self, 'history_df') and not self.history_df.empty:
               metrics = AnalyticsEngine.calculate_metrics(self.history_df)
               
            # Load trades for Expectancy
            trades_path = os.path.join(os.path.dirname(self.status_path), "trades.csv")
            if os.path.exists(trades_path):
                trades_df = pd.read_csv(trades_path)
                exp_stats = AnalyticsEngine.calculate_expectancy(trades_df)
                metrics.update(exp_stats)
        except Exception as e:
            logger.warning(f"⚠️ Analytics pre-calc failed: {e}")
        
        # 1. Prepare JSON Data (Structured Schema)
        json_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': session_id,
            'total_equity': equity,
            'cash': self.current_cash,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'positions': self.positions,
            'performance_metrics': metrics,  # New Phase 5 Field
            'math_stats': getattr(self, 'math_stats', {}), # New Phase 6 Field
            'strategy_rankings': getattr(self, 'strategy_rankings', {}), # Phase 7 Meta-Brain
            'global_regime': getattr(self, 'global_regime', 'UNKNOWN'), # Phase 8 Leader
            'global_regime_data': getattr(self, 'global_regime_data', {}), # Phase 8.1 Breadth
            'last_heartbeat': datetime.now(timezone.utc).isoformat()
        }
        
        # 2. Atomic Save via DataHandler
        handler = get_data_handler()
        # Phase 6 Fix: Use consistent name 'live_status.json' for dashboard prioritization
        json_path = os.path.join(os.path.dirname(self.status_path), "live_status.json")
        handler.save_live_status(json_path, json_data)
        
        # 3. Save CSV (Legacy Support & Chart History)
        # Flatten dicts for CSV
        csv_data = {
            'timestamp': datetime.now(timezone.utc),
            'total_equity': equity,
            'cash': self.current_cash,
            'realized_pnl': self.realized_pnl,
            'positions': json.dumps(self.positions),
            'strategy_performance': json.dumps(self.strategy_performance),
            'session_id': session_id
        }
        
        # APPEND THROTTLING (Optimization Phase 2)
        # Writing to CSV on every tick is too heavy. Only write CSV every 10s or critical events.
        # However, live_status.json needs to be relatively fresh for UI (1s is fine).
        
        now = datetime.now()
        if not hasattr(self, '_last_csv_save'): self._last_csv_save = datetime.min
        
        if (now - self._last_csv_save).total_seconds() > 5.0:
            self._last_csv_save = now
            # Append to status history
            df = pd.DataFrame([csv_data])
            header = not os.path.exists(self.status_path)
            try:
                df.to_csv(self.status_path, mode='a', header=header, index=False)
            except Exception as e:
                logger.error(f"⚠️ Failed to save status CSV: {e}")
            
        # 4. Session Copy (Optional)
        session_path = session_mgr.get_session_path() if session_mgr else None
        if session_path and session_path != os.path.dirname(self.status_path):
            try:
                # Also save JSON to session
                session_json = os.path.join(session_path, "live_status.json")
                handler.save_live_status(session_json, json_data)
                
                # And CSV
                session_csv = os.path.join(session_path, "status.csv")
                s_header = not os.path.exists(session_csv)
                df.to_csv(session_csv, mode='a', header=s_header, index=False)
            except:
                pass


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
        from datetime import datetime
        
        for symbol, position in self.positions.items():
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
                    print(f"🛑 STOP LOSS triggered for {symbol}: {pnl_pct*100:.2f}%")
                    from datetime import timezone
                    events_queue.put(SignalEvent("99", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0))
                    
                elif pnl_pct > 0.008:  # Take Profit: +0.8%
                    print(f"💰 TAKE PROFIT triggered for {symbol}: {pnl_pct*100:.2f}%")
                    from datetime import timezone
                    events_queue.put(SignalEvent("99", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0))
                    
                elif drawdown_from_peak < -0.002 and pnl_pct > 0.002:  # Trailing Stop: -0.2% from peak (only if in profit)
                    print(f"📉 TRAILING STOP triggered for {symbol}: Peak {hwm:.4f}, Now {current_price:.4f}")
                    from datetime import timezone
                    events_queue.put(SignalEvent("99", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0))
                    
            elif quantity < 0:  # SHORT position
                pnl_pct = (entry_price - current_price) / entry_price
                lwm = position.get('low_water_mark', current_price)
                drawup_from_low = (lwm - current_price) / lwm
                
                # SHORT Exit Conditions
                if pnl_pct < -0.003:  # Stop Loss: -0.3%
                    print(f"🛑 STOP LOSS triggered for SHORT {symbol}: {pnl_pct*100:.2f}%")
                    from datetime import timezone
                    events_queue.put(SignalEvent("99", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0))
                    
                elif pnl_pct > 0.008:  # Take Profit: +0.8%
                    print(f"💰 TAKE PROFIT triggered for SHORT {symbol}: {pnl_pct*100:.2f}%")
                    from datetime import timezone
                    events_queue.put(SignalEvent("99", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0))
                    
                elif drawup_from_low < -0.002 and pnl_pct > 0.002:  # Trailing Stop
                    print(f"📈 TRAILING STOP triggered for SHORT {symbol}: Low {lwm:.4f}, Now {current_price:.4f}")
                    from datetime import timezone
                    events_queue.put(SignalEvent("99", symbol, datetime.now(timezone.utc), SignalType.EXIT, strength=1.0))

    def log_trade_report(self, event, pnl=None, fill_price=0):
        """
        Prints a real-time report of the trade execution, Win Rate, and Balance.
        Requested by User: "cada que compre y venda me vaya informando sobre el winrate... y pnl y balance"
        """
        try:
            # 1. Global Performance Stats
            total_wins = 0
            total_losses = 0
            total_trades = 0
            
            for strat, data in self.strategy_performance.items():
                total_wins += data['wins']
                total_losses += data['losses']
                total_trades += data['trades']
            
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            
            # 2. Balance Stats
            equity = self.get_total_equity()
            balance_delta = equity - self.initial_capital
            balance_pct = (balance_delta / self.initial_capital) * 100
            
            # 3. Formatting
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
            print(f"⚠️ Report Error: {e}")

    def log_to_csv(self, data):
        df = pd.DataFrame([data])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        # logger.info(f"Logged: {data}")

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
