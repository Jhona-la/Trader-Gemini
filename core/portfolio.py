import pandas as pd
import os
import json
from datetime import datetime
from config import Config  # Import Config for Futures/Spot detection
from data.database import DatabaseHandler
from utils.logger import logger

class Portfolio:
    def __init__(self, initial_capital=10000.0, csv_path="dashboard/data/trades.csv", status_path="dashboard/data/status.csv", auto_save=True):
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
        self._cash_lock = threading.Lock()
        
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
        if event.type == 'MARKET':
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
        
        if self.auto_save:
            self.save_status()
            
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

    def update_signal(self, event):
        if event.type == 'SIGNAL':
            pass

    def update_fill(self, event):
        if event.type == 'FILL':
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
            
            # Deduct fee from Cash immediately (Fees are paid on every trade)
            with self._cash_lock:
                self.current_cash -= estimated_fee
            
            # Deduct fee from Cash immediately (Fees are paid on every trade)
            with self._cash_lock:
                self.current_cash -= estimated_fee
            
            logger.info(f"  💸 Fee Paid: ${estimated_fee:.4f} ({fee_rate*100}%)")
            
            if event.direction == 'BUY':
                # BUY can be: Close SHORT or Open LONG
                pos = self.positions.get(event.symbol, {'quantity': 0, 'avg_price': 0})
                
                if pos['quantity'] < 0:
                    # === CLOSING SHORT POSITION (Buying back) ===
                    # PnL = (entry_price - exit_price) * quantity
                    entry_price = pos['avg_price']
                    exit_price = fill_price
                    pnl = (entry_price - exit_price) * event.quantity
                    
                    self.realized_pnl += pnl
                    self.current_cash += pnl # Add PnL to cash
                    
                    # Release Margin
                    if Config.BINANCE_USE_FUTURES:
                        # Approximate margin release (proportional to closed qty)
                        # We assume margin was tracked correctly on open
                        # Ideally we'd track margin per position, but global is okay for now
                        self.used_margin = max(0, self.used_margin - margin_impact)
                    
                    # Update position
                    pos['quantity'] += event.quantity  # Add to negative (closes it)
                    
                    # Strategy performance tracking
                    strat_id = getattr(event, 'strategy_id', None) or pos.get('opener_strategy_id', 'Unknown')
                    if strat_id not in self.strategy_performance:
                        self.strategy_performance[strat_id] = {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0}
                    
                    self.strategy_performance[strat_id]['pnl'] += pnl
                    self.strategy_performance[strat_id]['trades'] += 1
                    if pnl > 0:
                        self.strategy_performance[strat_id]['wins'] += 1
                    elif pnl < 0:
                        self.strategy_performance[strat_id]['losses'] += 1
                    
                    if pnl > 0:
                        self.strategy_performance[strat_id]['wins'] += 1
                    elif pnl < 0:
                        self.strategy_performance[strat_id]['losses'] += 1
                    
                    logger.info(f"📈 SHORT Closed: {event.symbol} PnL=${pnl:.2f} (Entry: ${entry_price:.4f}, Exit: ${exit_price:.4f})")
                    
                    # FIXED: Value-based dust detection (consistent with close_position method)
                    # Check if remaining position is worth less than $1
                    position_value = abs(pos['quantity']) * pos.get('current_price', exit_price)
                    if position_value < 1.0:  # Less than $1 = dust
                        pos['quantity'] = 0
                        pos['low_water_mark'] = 0
                        pos['stop_distance'] = 0
                    
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
                
            elif event.direction == 'SELL':
                # SELL can be: Close LONG or Open SHORT
                pos = self.positions.get(event.symbol, {'quantity': 0, 'avg_price': 0})
                
                if pos['quantity'] > 0:
                    # === CLOSING LONG POSITION ===
                    
                    # Calculate Realized PnL
                    cost_basis = event.quantity * pos['avg_price']
                    pnl = fill_cost - cost_basis
                    self.realized_pnl += pnl
                    
                    # Update Cash/Margin
                    if Config.BINANCE_USE_FUTURES:
                        self.current_cash += pnl # Add PnL
                        self.used_margin = max(0, self.used_margin - margin_impact) # Release Margin
                    else:
                        self.current_cash += fill_cost # Spot: Get full cash back
                    
                    # Update Strategy Performance
                    strat_id = getattr(event, 'strategy_id', None)
                    if not strat_id and 'opener_strategy_id' in pos:
                        strat_id = pos['opener_strategy_id']
                    if not strat_id:
                        strat_id = 'Unknown'

                    if strat_id not in self.strategy_performance:
                        self.strategy_performance[strat_id] = {'pnl': 0.0, 'wins': 0, 'losses': 0, 'trades': 0}
                    
                    self.strategy_performance[strat_id]['pnl'] += pnl
                    self.strategy_performance[strat_id]['trades'] += 1
                    if pnl > 0:
                        self.strategy_performance[strat_id]['wins'] += 1
                    elif pnl < 0:
                        self.strategy_performance[strat_id]['losses'] += 1
                        
                    if pnl > 0:
                        self.strategy_performance[strat_id]['wins'] += 1
                    elif pnl < 0:
                        self.strategy_performance[strat_id]['losses'] += 1
                        
                    logger.info(f"💰 LONG Closed: {event.symbol} PnL=${pnl:.2f} (Strategy: {strat_id})")
                    
                    pos['quantity'] -= event.quantity
                    
                    # FIXED: Value-based dust detection (consistent with close_position method)
                    position_value = pos['quantity'] * pos.get('current_price', fill_price)
                    if position_value < 1.0:  # Less than $1 = dust
                        pos['quantity'] = 0
                        pos['high_water_mark'] = 0
                        pos['stop_distance'] = 0
                
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
                    
                    pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                    
                    logger.info(f"📉 SHORT Opened: {event.symbol} @ ${fill_price:.4f} (Qty: {event.quantity})")
            
            # Log Trade
            self.log_to_csv({
                'datetime': datetime.now(),
                'symbol': event.symbol,
                'type': 'FILL',
                'direction': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'fill_cost': event.quantity * fill_price,
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                'details': f"Exchange: {event.exchange} | Margin: {margin_impact:.2f}"
            })
            
            # Log Trade to DB
            self.db.log_trade({
                'symbol': event.symbol,
                'side': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'timestamp': datetime.now(),
                'order_type': 'MARKET', # Assuming market for now
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),
                'pnl': 0.0, # PnL is calculated on close, but we log the trade execution here
                'commission': estimated_fee
            })
            
            # Update Position in DB
            pos = self.positions[event.symbol]
            self.db.update_position(
                symbol=event.symbol,
                quantity=pos['quantity'],
                entry_price=pos['avg_price'],
                current_price=fill_price,
                pnl=0.0 # Reset PnL on new trade execution until price update
            )
            
            if self.auto_save:
                self.save_status()

    def save_status(self):
        """
        Save current portfolio state to CSV for Dashboard AND JSON for Crash Recovery.
        """
        equity = self.get_total_equity()
        unrealized_pnl = equity - self.initial_capital - self.realized_pnl
        
        # 1. Prepare State Data
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'total_equity': equity,
            'cash': self.current_cash,
            'used_margin': self.used_margin,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'positions': self.positions, # Raw dict
            'strategy_performance': self.strategy_performance # Raw dict
        }
        
        # 2. Save Atomic JSON (For Recovery)
        # We save to a dedicated JSON file that matches what load_portfolio_state expects
        json_path = self.status_path.replace('.csv', '.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(state_data, f, indent=4)
        except Exception as e:
            print(f"⚠️ Failed to save portfolio state JSON: {e}")
        
        # 3. Save CSV (For Dashboard History)
        # Flatten dicts for CSV
        csv_data = state_data.copy()
        csv_data['timestamp'] = datetime.now() # Use datetime object for pandas
        csv_data['positions'] = json.dumps(self.positions)
        csv_data['strategy_performance'] = json.dumps(self.strategy_performance)
        
        # Append to status history
        df = pd.DataFrame([csv_data])
        header = not os.path.exists(self.status_path)
        try:
            df.to_csv(self.status_path, mode='a', header=header, index=False)
        except Exception as e:
            print(f"⚠️ Failed to save status CSV: {e}")


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
                    events_queue.put(SignalEvent(99, symbol, datetime.now(), 'EXIT', reason='stop_loss'))
                    
                elif pnl_pct > 0.008:  # Take Profit: +0.8%
                    print(f"💰 TAKE PROFIT triggered for {symbol}: {pnl_pct*100:.2f}%")
                    events_queue.put(SignalEvent(99, symbol, datetime.now(), 'EXIT', reason='take_profit'))
                    
                elif drawdown_from_peak < -0.002 and pnl_pct > 0.002:  # Trailing Stop: -0.2% from peak (only if in profit)
                    print(f"📉 TRAILING STOP triggered for {symbol}: Peak {hwm:.4f}, Now {current_price:.4f}")
                    events_queue.put(SignalEvent(99, symbol, datetime.now(), 'EXIT', reason='trailing_stop'))
                    
            elif quantity < 0:  # SHORT position
                pnl_pct = (entry_price - current_price) / entry_price
                lwm = position.get('low_water_mark', current_price)
                drawup_from_low = (lwm - current_price) / lwm
                
                # SHORT Exit Conditions
                if pnl_pct < -0.003:  # Stop Loss: -0.3%
                    print(f"🛑 STOP LOSS triggered for SHORT {symbol}: {pnl_pct*100:.2f}%")
                    events_queue.put(SignalEvent(99, symbol, datetime.now(), 'EXIT', reason='stop_loss'))
                    
                elif pnl_pct > 0.008:  # Take Profit: +0.8%
                    print(f"💰 TAKE PROFIT triggered for SHORT {symbol}: {pnl_pct*100:.2f}%")
                    events_queue.put(SignalEvent(99, symbol, datetime.now(), 'EXIT', reason='take_profit'))
                    
                elif drawup_from_low < -0.002 and pnl_pct > 0.002:  # Trailing Stop
                    print(f"📈 TRAILING STOP triggered for SHORT {symbol}: Low {lwm:.4f}, Now {current_price:.4f}")
                    events_queue.put(SignalEvent(99, symbol, datetime.now(), 'EXIT', reason='trailing_stop'))

    def log_to_csv(self, data):
        df = pd.DataFrame([data])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        # logger.info(f"Logged: {data}")
