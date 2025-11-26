import pandas as pd
import os
import json
from datetime import datetime

class Portfolio:
    def __init__(self, initial_capital=10000.0, csv_path="dashboard/data/trades.csv", status_path="dashboard/data/status.csv", auto_save=True):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.pending_cash = 0.0  # Cash reserved for pending orders
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
                # If cleanly shut down, we might not want to load positions if they were closed
                # But if they weren't closed (crash), we need them.
                # We'll trust the 'positions' field.
                pass
                
            # Restore Cash & PnL
            self.current_cash = data.get('cash', self.initial_capital)
            self.realized_pnl = data.get('realized_pnl', 0.0)
            
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
        """Return cash available for trading (excluding pending orders)."""
        with self._cash_lock:
            return self.current_cash - self.pending_cash
    
    def reserve_cash(self, amount):
        """Reserve cash for a pending order. Returns True if successful."""
        with self._cash_lock:
            if self.current_cash - self.pending_cash >= amount:
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
            # In a real system, we'd get price from the event. 
            # For now, we assume the event carries the latest bar.
            # But MarketEvent in this system is just a trigger.
            # We rely on DataHandler to have updated bars.
            # This part is tricky without direct access to DataHandler here.
            # We will rely on the Engine passing a price-enriched event or fetching it.
            pass 
            
    def update_market_price(self, symbol, price):
        """
        Helper to update current price of a symbol for PnL calculation.
        Also updates High Water Mark for Trailing Stops.
        """
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0, 'current_price': price, 'high_water_mark': price, 'stop_distance': 0}
        else:
            self.positions[symbol]['current_price'] = price
            # Update High Water Mark if we have a position
            if self.positions[symbol]['quantity'] > 0:
                if price > self.positions[symbol].get('high_water_mark', 0):
                    self.positions[symbol]['high_water_mark'] = price
        
        if self.auto_save:
            self.save_status()

    def update_signal(self, event):
        """
        Receive signal event for optional tracking.
        NOTE: Signals are NOT trades - don't log to trades CSV.
        Only FILLS are actual trades.
        """
        if event.type == 'SIGNAL':
            # Optional: Could log to separate signals log if needed
            # For now, just pass - engine handles signal->order->fill flow
            pass

    def update_fill(self, event):
        if event.type == 'FILL':
            # Update Cash and Positions
            fill_cost = event.fill_cost # Total cost (price * quantity)
            fill_price = event.fill_cost / event.quantity if event.quantity > 0 else 0
            
            if event.direction == 'BUY':
                # Release pending cash (it's now confirmed spent)
                with self._cash_lock:
                    self.current_cash -= fill_cost
                    self.pending_cash = max(0, self.pending_cash - fill_cost)
                
                # Update Avg Price
                if event.symbol not in self.positions:
                    self.positions[event.symbol] = {
                        'quantity': 0, 
                        'avg_price': 0, 
                        'current_price': 0, 
                        'high_water_mark': 0, 
                        'stop_distance': 0,
                        'opener_strategy_id': getattr(event, 'strategy_id', None) # Track who opened it
                    }
                
                pos = self.positions[event.symbol]
                # If opening new position (was 0), update opener
                if pos['quantity'] == 0:
                     pos['opener_strategy_id'] = getattr(event, 'strategy_id', None)
                
                total_cost = (pos['quantity'] * pos['avg_price']) + fill_cost
                total_qty = pos['quantity'] + event.quantity
                
                if total_qty > 0:
                    pos['avg_price'] = total_cost / total_qty
                pos['quantity'] = total_qty
                
                # PYRAMIDING FIX: Update HWM only if new fill is higher
                # Don't reset HWM when adding to existing position
                pos['high_water_mark'] = max(pos.get('high_water_mark', 0), fill_price)
                
            elif event.direction == 'SELL':
                self.current_cash += fill_cost
                
                # Calculate Realized PnL
                pos = self.positions.get(event.symbol, {'quantity': 0, 'avg_price': 0})
                cost_basis = event.quantity * pos['avg_price']
                pnl = fill_cost - cost_basis
                self.realized_pnl += pnl
                
                # Update Strategy Performance
                strat_id = getattr(event, 'strategy_id', None)
                
                # If strategy_id is missing (Risk Manager or Manual Close), use the opener
                if not strat_id and 'opener_strategy_id' in pos:
                    strat_id = pos['opener_strategy_id']
                    print(f"ℹ️  Attributing Risk/Manual Exit to Strategy {strat_id}")
                
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
                    
                print(f"💰 PnL Update: Strategy {strat_id} PnL: ${pnl:.2f} (Total: ${self.strategy_performance[strat_id]['pnl']:.2f})")
                
                pos['quantity'] -= event.quantity
                # Avg price doesn't change on sell (FIFO/Weighted Avg assumption)
                
                # Reset HWM if closed
                if pos['quantity'] <= 0.00001: # Float tolerance
                    pos['quantity'] = 0
                    pos['high_water_mark'] = 0
                    pos['stop_distance'] = 0
            
            # Log Trade
            self.log_to_csv({
                'datetime': datetime.now(), # Use current time for fill
                'symbol': event.symbol,
                'type': 'FILL',
                'direction': event.direction,
                'quantity': event.quantity,
                'price': fill_price,
                'fill_cost': event.quantity * fill_price,
                'strategy_id': getattr(event, 'strategy_id', 'Unknown'),  # Extract from FillEvent
                'details': f"Exchange: {event.exchange}"
            })
            
            if self.auto_save:
                self.save_status()

    def get_total_equity(self):
        equity = self.current_cash
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
