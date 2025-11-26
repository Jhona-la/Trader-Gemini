from core.events import OrderEvent
from config import Config

class RiskManager:
    """
    Risk Management Module.
    Responsible for:
    1. Checking if a trade is allowed (Risk Limits).
    2. Sizing the trade (Position Sizing).
    3. Attaching Stop Loss / Take Profit orders.
    """
    def __init__(self, max_concurrent_positions=5, portfolio=None):
        self.max_risk_per_trade = Config.MAX_RISK_PER_TRADE # e.g. 1%
        self.stop_loss_pct = Config.STOP_LOSS_PCT       # e.g. 2%
        self.current_capital = 10000.0 # Mock capital for now
        self.max_concurrent_positions = max_concurrent_positions
        self.portfolio = portfolio  # Reference for balance checks
        
        # STRATEGY COORDINATION FIX #2: Cooldown Mechanism
        # Prevents re-entering a trade immediately after exit (churn prevention)
        # Format: {symbol: timestamp_until_allowed}
        self.cooldowns = {}
        # OPTIMIZED: Faster cycling (was 15)
        self.base_cooldown_minutes = 5
        self.current_regime = 'RANGING' # Default, updated by main.py

    def size_position(self, signal_event, current_price):
        """
        Calculate the quantity to trade based on risk per trade.
        Risk Amount = Capital * Max Risk %
        Position Size = Risk Amount / (Price * Stop Loss %)
        """
        # Default: 15% of capital per trade (Aggressive sizing for crypto)
        target_exposure = self.current_capital * 0.15 
        
        # Volatility Sizing (ATR)
        if hasattr(signal_event, 'atr') and signal_event.atr is not None and signal_event.atr > 0:
            risk_amount = self.current_capital * self.max_risk_per_trade
            stop_distance = signal_event.atr * 2.0
        
        # Adjust by signal strength (Kelly Criterion-lite)
        if hasattr(signal_event, 'strength'):
            target_exposure *= signal_event.strength
            
        return target_exposure

    def generate_order(self, signal_event, current_price):
        """
        Converts a SignalEvent into a verified OrderEvent.
        Includes balance checking and cash reservation.
        """
        # 0. Check max concurrent positions
        if self.portfolio:
            open_positions = sum(1 for pos in self.portfolio.positions.values() if pos['quantity'] != 0)
            if open_positions >= self.max_concurrent_positions and signal_event.signal_type == 'LONG':
                print(f"⚠️  Risk Manager: Max {self.max_concurrent_positions} positions reached. Signal rejected.")
                return None
        
        # 0.5 Check Cooldowns (Churn Prevention)
        if signal_event.signal_type == 'LONG':
            if signal_event.symbol in self.cooldowns:
                if signal_event.datetime < self.cooldowns[signal_event.symbol]:
                    print(f"❄️  Risk Manager: Cooldown active for {signal_event.symbol}. Skipping LONG.")
                    return None
                else:
                    # Cooldown expired
                    del self.cooldowns[signal_event.symbol]
        
        # PYRAMIDING: Allow adding to winning positions
        # Check if we already have a position and if it's profitable
        if self.portfolio and signal_event.signal_type == 'LONG':
            existing = self.portfolio.positions.get(signal_event.symbol, {})
            existing_qty = existing.get('quantity', 0)
            
            if existing_qty > 0:
                # We already have a position - check if we can pyramid
                current_price = current_price
                entry_price = existing.get('avg_price', 0)
                
                if entry_price > 0:
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # OPTIMIZED: Lower threshold (was 1.5%)
                    if profit_pct >= 0.8:
                        # Position is profitable (+0.8%), allow pyramiding (Aggressive)
                        # Reduce signal strength to 50% to limit max size
                        original_strength = getattr(signal_event, 'strength', 1.0)
                        signal_event.strength = original_strength * 0.5
                        print(f"📈 PYRAMIDING: Adding to {signal_event.symbol} at +{profit_pct:.1f}% profit (50% size)")
                    else:
                        print(f"⚠️  Risk Manager: Already have position in {signal_event.symbol} (+{profit_pct:.1f}% profit, need +0.8% for pyramiding). Duplicate LONG rejected.")
                        return None
                else:
                    print(f"⚠️  Risk Manager: Already have position in {signal_event.symbol}. Duplicate LONG rejected.")
                    return None
        
        # 1. Determine Quantity based on Signal Type
        if signal_event.signal_type in ['EXIT', 'SHORT']:
            # EXIT/SHORT means close existing position
            # We must sell the EXACT quantity we hold
            if self.portfolio and signal_event.symbol in self.portfolio.positions:
                quantity = self.portfolio.positions[signal_event.symbol]['quantity']
                dollar_size = quantity * current_price
                if quantity <= 0:
                    print(f"⚠️  Risk Manager: Ignore EXIT for {signal_event.symbol} - No position.")
                    return None
            else:
                print(f"⚠️  Risk Manager: Ignore EXIT for {signal_event.symbol} - Portfolio data missing.")
                return None
        else:
            # LONG means open new position -> Calculate Size
            dollar_size = self.size_position(signal_event, current_price)
            
            # Convert to Quantity (Units)
            if current_price == 0:
                return None
                
            quantity = dollar_size / current_price
            
            # Rounding (Crypto usually allows decimals, Stocks integer)
            if 'USDT' in signal_event.symbol:
                quantity = round(quantity, 5)
            else:
                quantity = int(quantity)
            
            if quantity <= 0:
                return None
        
        # 3. VALIDATE & RESERVE CASH (NEW)
        if self.portfolio and signal_event.signal_type == 'LONG':
            # BUY orders need cash
            available = self.portfolio.get_available_cash()
            
            # MINIMUM ORDER SIZE CHECK (Binance requires ~$5)
            if dollar_size < 5.0:
                print(f"⚠️  Risk Manager: Order size ${dollar_size:.2f} too small (Min $5). Skipping.")
                return None
                
            if available < dollar_size:
                print(f"⚠️  Risk Manager: Insufficient balance. Need ${dollar_size:.2f}, have ${available:.2f}")
                return None
            
            # Reserve cash for this order
            if not self.portfolio.reserve_cash(dollar_size):
                print(f"⚠️  Risk Manager: Failed to reserve ${dollar_size:.2f}")
                return None

        # 4. Create Order
        order_type = 'MKT'
        
        # Map signal type to order direction
        if signal_event.signal_type == 'LONG':
            direction = 'BUY'
        elif signal_event.signal_type in ['EXIT', 'SHORT']:
            # EXIT means close a long position = SELL
            direction = 'SELL'
            
            # ACTIVATE COOLDOWN ON EXIT
            # OPTIMIZED COOLDOWN: Faster re-entry for more opportunities
            cooldown_duration = self.base_cooldown_minutes
            if self.current_regime == 'TRENDING_BULL':
                cooldown_duration = 3   # Very aggressive in bull runs (was 5)
            elif self.current_regime == 'CHOPPY':
                cooldown_duration = 10  # Moderate in chop (was 30)
            elif self.current_regime == 'TRENDING_BEAR':
                cooldown_duration = 20  # Conservative in bear (was 60)
                
            from datetime import timedelta
            self.cooldowns[signal_event.symbol] = signal_event.datetime + timedelta(minutes=cooldown_duration)
            print(f"❄️  Risk Manager: Cooldown activated for {signal_event.symbol} ({cooldown_duration}m) until {self.cooldowns[signal_event.symbol]}")
        else:
            print(f"⚠️  Risk Manager: Unknown signal type: {signal_event.signal_type}")
            return None
        
        print(f"✅ Risk Manager: Approved {direction} {quantity} {signal_event.symbol} (${dollar_size:.2f})")
        
        # Pass strategy_id to OrderEvent
        strategy_id = getattr(signal_event, 'strategy_id', None)
        return OrderEvent(signal_event.symbol, order_type, quantity, direction, strategy_id)


    def check_stops(self, portfolio, data_provider):
        """
        SMART TRAILING STOP + TAKE PROFIT LEVELS:
        - TP1 (+1%): Lock profits early
        - TP2 (+2%): Tighter trailing (25% of gain)
        - TP3 (+3%+): Very tight trailing (10% of gain)
        - Stop Loss: 2% below entry if not profitable
        
        Returns a list of SignalEvents (EXIT) if stops/TPs are triggered.
        """
        stop_signals = []
        from core.events import SignalEvent
        from datetime import datetime
        
        for symbol, pos in portfolio.positions.items():
            qty = pos['quantity']
            if qty <= 0:
                continue
                
            current_price = pos['current_price']
            entry_price = pos['avg_price']
            hwm = pos.get('high_water_mark', entry_price)
            
            if current_price == 0 or entry_price == 0:
                continue
            
            # Calculate unrealized profit %
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # TAKE PROFIT LEVELS SYSTEM
            if unrealized_pnl_pct >= 3.0:
                # TP3: Very tight trailing (10% of gain from HWM)
                gain_from_entry = hwm - entry_price
                trail_distance = gain_from_entry * 0.1  # Very tight
                stop_price = hwm - trail_distance
                
                if current_price < stop_price:
                    print(f"💰 TP3 Hit for {symbol}! Price: {current_price:.4f}, Entry: {entry_price:.4f}, HWM: {hwm:.4f} (Profit: +{unrealized_pnl_pct:.2f}%)")
                    sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                    stop_signals.append(sig)
                    
            elif unrealized_pnl_pct >= 2.0:
                # TP2: Tighter trailing (25% of gain from HWM)
                gain_from_entry = hwm - entry_price
                trail_distance = gain_from_entry * 0.25
                stop_price = hwm - trail_distance
                
                # Minimum at +1.5% to lock some profit
                min_stop = entry_price * 1.015
                stop_price = max(stop_price, min_stop)
                
                if current_price < stop_price:
                    print(f"💰 TP2 Hit for {symbol}! Price: {current_price:.4f}, Entry: {entry_price:.4f}, HWM: {hwm:.4f} (Profit: +{unrealized_pnl_pct:.2f}%)")
                    sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                    stop_signals.append(sig)
                    
            elif unrealized_pnl_pct >= 1.0:
                # TP1: Standard trailing (50% of gain from HWM)
                gain_from_entry = hwm - entry_price
                trail_distance = gain_from_entry * 0.5
                stop_price = hwm - trail_distance
                
                # Ensure stop is at least at breakeven + commission + slippage
                # Binance Fees: ~0.1% maker + 0.1% taker = 0.2% round trip
                # We set buffer to 0.3% to be safe
                breakeven_stop = entry_price * 1.003  # Breakeven + 0.3%
                stop_price = max(stop_price, breakeven_stop)
                
                if current_price < stop_price:
                    print(f"💰 TP1 Hit for {symbol}! Price: {current_price:.4f}, Entry: {entry_price:.4f}, HWM: {hwm:.4f} (Profit: +{unrealized_pnl_pct:.2f}%)")
                    sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                    stop_signals.append(sig)

            elif unrealized_pnl_pct >= 0.6:
                # MICRO-TREND CAPTURE (Scalping Mode)
                # If we are up +0.6%, don't let it go back to breakeven (+0.3%)
                # Lock in at least +0.4%
                stop_price = entry_price * 1.004
                
                if current_price < stop_price:
                    print(f"⚡ Micro-Scalp Hit for {symbol}! Price: {current_price:.4f}, Entry: {entry_price:.4f} (Profit: +{unrealized_pnl_pct:.2f}%)")
                    sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                    stop_signals.append(sig)
                    
            else:
                # Not yet profitable: Use standard stop loss
                stop_distance = pos.get('stop_distance', current_price * 0.02)  # 2% default
                stop_price = entry_price - stop_distance
                
                if current_price < stop_price:
                    print(f"🛑 Stop Loss Hit for {symbol}! Price: {current_price:.4f}, Entry: {entry_price:.4f}, Stop: {stop_price:.4f} (Loss: {unrealized_pnl_pct:.2f}%)")
                    sig = SignalEvent("RISK_MGR", symbol, datetime.now(), 'EXIT', strength=1.0)
                    stop_signals.append(sig)
                
        return stop_signals
