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
        # CRITICAL FIX: Use actual portfolio capital if available
        if self.portfolio:
            # Use Total Equity (Cash + Unrealized PnL) for sizing
            # This allows compounding gains and shrinking on losses
            capital = self.portfolio.get_total_equity()
        else:
            capital = self.current_capital # Fallback (should not happen in prod)

        # Default: 15% of capital per trade (Aggressive sizing for crypto)
        # Dynamic Scaling:
        # If Capital < 1000: Use 20% (to grow faster)
        # If Capital > 10000: Use 10% (to preserve wealth)
        if capital < 1000:
            base_pct = 0.20
        elif capital > 10000:
            base_pct = 0.10
        else:
            base_pct = 0.15
            
        target_exposure = capital * base_pct
        
        # Volatility Sizing (ATR)
        if hasattr(signal_event, 'atr') and signal_event.atr is not None and signal_event.atr > 0:
            # Risk 1% of capital per trade
            risk_amount = capital * self.max_risk_per_trade
            stop_distance = signal_event.atr * 2.0
            # Position = Risk / StopDistance
            # e.g. Risk $100, Stop $50 away -> Buy 2 units
            if stop_distance > 0:
                vol_adjusted_size = (risk_amount / stop_distance) * current_price
                # Cap at 2x target exposure to prevent massive leverage on low vol
                target_exposure = min(vol_adjusted_size, target_exposure * 2.0)
        
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
            # FIXED: Apply limit to BOTH Long and Short signals
            if open_positions >= self.max_concurrent_positions and signal_event.signal_type in ['LONG', 'SHORT']:
                print(f"⚠️  Risk Manager: Max {self.max_concurrent_positions} positions reached. Signal rejected.")
                return None
        
        # 0.5 Check Cooldowns (Churn Prevention)
        # FIXED: Apply cooldown to both directions
        if signal_event.signal_type in ['LONG', 'SHORT']:
            if signal_event.symbol in self.cooldowns:
                if signal_event.datetime < self.cooldowns[signal_event.symbol]:
                    print(f"❄️  Risk Manager: Cooldown active for {signal_event.symbol}. Skipping {signal_event.signal_type}.")
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
        
        # === EXIT: Close any existing position ===
        if signal_event.signal_type == 'EXIT':
            if self.portfolio and signal_event.symbol in self.portfolio.positions:
                existing_qty = self.portfolio.positions[signal_event.symbol]['quantity']
                
                if existing_qty == 0:
                    print(f"⚠️  Risk Manager: Ignore EXIT for {signal_event.symbol} - No position.")
                    return None
                
                # Determine direction based on current position
                quantity = abs(existing_qty)
                direction = 'SELL' if existing_qty > 0 else 'BUY'
                dollar_size = quantity * current_price
            else:
                print(f"⚠️  Risk Manager: Ignore EXIT for {signal_event.symbol} - Portfolio data missing.")
                return None
        
        # === LONG: Open long position ===
        elif signal_event.signal_type == 'LONG':
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
            
            direction = 'BUY'
        
        # === SHORT: Open short position ===
        elif signal_event.signal_type == 'SHORT':
            # CRITICAL CHECK: Spot Trading does not support "Shorting" (selling without owning)
            # unless using Margin Mode (Cross/Isolated) which requires borrowing.
            # For safety in this bot version, we BLOCK Shorting in Spot Mode.
            if not Config.BINANCE_USE_FUTURES:
                print(f"⚠️  Risk Manager: SHORT signal rejected. Spot Trading does not support direct shorting (requires Futures).")
                return None

            # Check max positions
            if self.portfolio:
                open_positions = sum(1 for pos in self.portfolio.positions.values() if pos['quantity'] != 0)
                if open_positions >= self.max_concurrent_positions:
                    print(f"⚠️  Risk Manager: Max {self.max_concurrent_positions} positions reached. SHORT rejected.")
                    return None
            
            # Check cooldown
            if signal_event.symbol in self.cooldowns:
                if signal_event.datetime < self.cooldowns[signal_event.symbol]:
                    print(f"❄️  Risk Manager: Cooldown active for {signal_event.symbol}. Skipping SHORT.")
                    return None
                else:
                    del self.cooldowns[signal_event.symbol]
            
            # Size the SHORT position (same as LONG)
            dollar_size = self.size_position(signal_event, current_price)
            
            if current_price == 0:
                return None
            
            quantity = dollar_size / current_price
            
            # Rounding
            if 'USDT' in signal_event.symbol:
                quantity = round(quantity, 5)
            else:
                quantity = int(quantity)
            
            if quantity <= 0:
                return None
            
            # In Futures, SELL opens SHORT
            direction = 'SELL'
        
        else:
            print(f"⚠️  Risk Manager: Unknown signal type: {signal_event.signal_type}")
            return None
        
        # 3. VALIDATE & RESERVE CASH (Both LONG and SHORT need margin in Futures)
        if self.portfolio and signal_event.signal_type in ['LONG', 'SHORT']:
            # Both BUY and SELL orders need cash/margin in Futures
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
                cooldown_duration = 2   # Very aggressive in bear runs
            elif self.current_regime == 'CHOPPY':
                cooldown_duration = 15  # Stay out longer in chop
            elif self.current_regime == 'RANGING':
                cooldown_duration = 5   # Standard for mean reversion
        else:
            # ENTRY Cooldown (Debounce): Prevent double-entry on rapid signals
            # 1 minute is enough to wait for fill and portfolio update
            cooldown_duration = 1
                
        from datetime import timedelta
        self.cooldowns[signal_event.symbol] = signal_event.datetime + timedelta(minutes=cooldown_duration)
        print(f"❄️  Risk Manager: Cooldown activated for {signal_event.symbol} ({cooldown_duration}m) until {self.cooldowns[signal_event.symbol]}")
        
        # 5. Create Order
        order_type = 'MKT'
        
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
            if qty == 0:  # Changed from <= to allow SHORT positions (qty < 0)
                continue
                
            current_price = pos['current_price']
            entry_price = pos['avg_price']
            
            if current_price == 0 or entry_price == 0:
                continue
            
            # ===============================
            # LONG POSITION (qty > 0)
            # ===============================
            if qty > 0:
                hwm = pos.get('high_water_mark', entry_price)
                
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
            
            # ===============================
            # SHORT POSITION (qty < 0)
            # ===============================
            elif qty < 0:
                lwm = pos.get('low_water_mark', entry_price)
                
                # For SHORT: profit when price goes DOWN
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                # TAKE PROFIT LEVELS SYSTEM FOR SHORT
                if unrealized_pnl_pct >= 3.0:
                    # TP3: Very tight trailing (10% of gain from LWM)
                    gain_from_entry = entry_price - lwm
                    trail_distance = gain_from_entry * 0.1
                    stop_price = lwm + trail_distance  # Price rising from LWM
                    
                    if current_price > stop_price:
                        print(f"💰 SHORT TP3 Hit for {symbol}! Price: {current_price:.4f}, Entry: {entry_price:.4f}, LWM: {lwm:.4f} (Profit: +{unrealized_pnl_pct:.2f}%)")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                        stop_signals.append(sig)
                
                elif unrealized_pnl_pct >= 2.0:
                    # TP2: Tighter trailing (25% of gain)
                    gain_from_entry = entry_price - lwm
                    trail_distance = gain_from_entry * 0.25
                    stop_price = lwm + trail_distance
                    min_stop = entry_price * 0.985  # Lock at least 1.5% profit
                    stop_price = min(stop_price, min_stop)
                    
                    if current_price > stop_price:
                        print(f"💰 SHORT TP2 Hit for {symbol}! Profit: +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                        stop_signals.append(sig)
                
                elif unrealized_pnl_pct >= 1.0:
                    # TP1: Standard trailing (50% of gain)
                    gain_from_entry = entry_price - lwm
                    trail_distance = gain_from_entry * 0.5
                    stop_price = lwm + trail_distance
                    breakeven_stop = entry_price * 0.997  # Breakeven - 0.3%
                    stop_price = min(stop_price, breakeven_stop)
                    
                    if current_price > stop_price:
                        print(f"💰 SHORT TP1 Hit for {symbol}! Profit: +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                        stop_signals.append(sig)
                
                elif unrealized_pnl_pct >= 0.6:
                    # Micro-scalp (0.6% profit, lock +0.4%)
                    stop_price = entry_price * 0.996  # Lock +0.4%
                    
                    if current_price > stop_price:
                        print(f"⚡ SHORT Micro-Scalp Hit for {symbol}! Profit: +{unrealized_pnl_pct:.2f}%")
                        sig = SignalEvent("TP_MANAGER", symbol, datetime.now(), 'EXIT', strength=1.0)
                        stop_signals.append(sig)
                
                else:
                    # Stop Loss (not profitable, price rising = loss for SHORT)
                    stop_distance = pos.get('stop_distance', current_price * 0.02)
                    stop_price = entry_price + stop_distance  # Price rising = loss
                    
                    if current_price > stop_price:
                        print(f"🛑 SHORT Stop Loss Hit for {symbol}! Price: {current_price:.4f}, Entry: {entry_price:.4f} (Loss: {unrealized_pnl_pct:.2f}%)")
                        sig = SignalEvent("RISK_MGR", symbol, datetime.now(), 'EXIT', strength=1.0)
                        stop_signals.append(sig)
                
        return stop_signals
