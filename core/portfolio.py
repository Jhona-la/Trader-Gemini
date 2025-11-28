import pandas as pd
import os
import json
from datetime import datetime
from config import Config  # Import Config for Futures/Spot detection

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
            
            print(f"  💸 Fee Paid: ${estimated_fee:.4f} ({fee_rate*100}%)")
            
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
                    
                    print(f"📈 SHORT Closed: {event.symbol} PnL=${pnl:.2f} (Entry: ${entry_price:.4f}, Exit: ${exit_price:.4f})")
                    
                    # Reset LWM if fully closed
                    if abs(pos['quantity']) < 0.00001:
                        pos['quantity'] = 0
                        pos['low_water_mark'] = 0
                        pos['stop_distance'] = 0
                    
                    # Release Pending Cash (Margin was reserved)
                    with self._cash_lock:
                        # In Futures, we reserved 'dollar_size' (margin) in RiskManager?
                        # RiskManager reserves 'dollar_size' which is Notional / Leverage?
                        # No, RiskManager reserves Full Notional usually?
                        # Let's check RiskManager. It reserves 'dollar_size'.
                        # If RiskManager reserves Margin, we release Margin.
                        # If RiskManager reserves Notional, we release Notional.
                        # Assuming RiskManager reserves Margin for Futures (we should verify this).
                        # For now, we release whatever was pending.
                        self.pending_cash = max(0, self.pending_cash - margin_impact) # Release the margin we reserved
                
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
                        
                    print(f"💰 LONG Closed: {event.symbol} PnL=${pnl:.2f} (Strategy: {strat_id})")
                    
                    pos['quantity'] -= event.quantity
                    
                    if pos['quantity'] <= 0.00001:
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
                    
                    print(f"📉 SHORT Opened: {event.symbol} @ ${fill_price:.4f} (Qty: {event.quantity})")
            
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
            
            if self.auto_save:
                self.save_status()

    def get_total_equity(self):
        equity = self.current_cash
        # In Futures, Equity = Cash + Unrealized PnL
        # In Spot, Equity = Cash + Market Value of Assets
        
        if Config.BINANCE_USE_FUTURES:
            # Futures Equity = Cash Balance + Unrealized PnL of all positions
            unrealized_pnl = 0.0
            for symbol, pos in self.positions.items():
                if pos['quantity'] != 0:
                    entry = pos['avg_price']
                    curr = pos['current_price']
                    qty = pos['quantity']
                    
                    if qty > 0: # LONG
                        unrealized_pnl += (curr - entry) * qty
                    else: # SHORT
                        unrealized_pnl += (entry - curr) * abs(qty)
            return equity + unrealized_pnl
        else:
            # Spot Equity
            for symbol, pos in self.positions.items():
                equity += pos['quantity'] * pos['current_price']
            return equity

    def save_status(self):
        """
        Save current portfolio state to CSV for Dashboard.
        """
        equity = self.get_total_equity()
        unrealized_pnl = equity - self.initial_capital - self.realized_pnl
        
        status = {
            'timestamp': datetime.now(),
            'total_equity': equity,
            'cash': self.current_cash,
            'used_margin': self.used_margin, # Log margin
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'positions': json.dumps(self.positions),
            'strategy_performance': json.dumps(self.strategy_performance)
        }
        
        # Append to status history
        df = pd.DataFrame([status])
        header = not os.path.exists(self.status_path)
        df.to_csv(self.status_path, mode='a', header=header, index=False)

    def log_to_csv(self, data):
        df = pd.DataFrame([data])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        print(f"Logged: {data}")
