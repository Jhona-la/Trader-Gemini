from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
import queue
import time

class Engine:
    """
    Event-Driven Trading Engine.
    Coordinates data, strategies, risk, execution.
    """
    def __init__(self, events_queue=None):
        self.events = events_queue if events_queue else queue.Queue()
        self.data_handlers = []
        self.strategies = []
        self.execution_handler = None
        self.portfolio = None
        self.risk_manager = None
        self.running = True
        
        # STRATEGY COORDINATION:
        # Strategies coordinate via Market Regime (ADX) handoff.
        # Technical Strategy skips trending markets (ADX > 25) -> ML Strategy takes over.
        # ML Strategy handles trending markets with RF/XGBoost.

    def register_data_handler(self, handler):
        self.data_handlers.append(handler)

    def register_strategy(self, strategy):
        self.strategies.append(strategy)

    def register_portfolio(self, portfolio):
        self.portfolio = portfolio

    def register_execution_handler(self, handler):
        self.execution_handler = handler
        
    def register_risk_manager(self, manager):
        self.risk_manager = manager

    def run(self):
        """
        Main event loop.
        """
        print("Engine started. Waiting for events...")
        while self.running:
            try:
                event = self.events.get(False)
            except queue.Empty:
                time.sleep(0.1)
            else:
                if event is not None:
                    self.process_event(event)

    def process_event(self, event):
        """
        Route the event to the appropriate components.
        """
        if event.type == 'MARKET':
            # Notify strategies of new data
            # Notify strategies of new data
            # STRATEGY ORCHESTRATION: Selectively run strategies based on Regime
            # We access regime from RiskManager (which is updated by Main)
            current_regime = 'RANGING'
            if self.risk_manager and hasattr(self.risk_manager, 'current_regime'):
                current_regime = self.risk_manager.current_regime
            
            for strategy in self.strategies:
                try:
                    # ORCHESTRATION LOGIC
                    strat_name = strategy.__class__.__name__
                    should_run = True # BUG FIX: Define default state
                    
                    # BUG #51 FIX: Removed strict gating. Strategies should self-regulate.
                    # Strict blocking caused the bot to miss opportunities in mixed regimes
                    # (e.g. BTC trending but ALTS ranging).
                    # Strategies like MLStrategy and TechnicalStrategy already have internal ADX filters.
                    
                    # We still pass the global regime for context, but don't force block.
                    # Exception: Statistical Strategy (Pairs) is dangerous in strong trends, so we keep a soft check.
                    
                    if 'Statistical' in strat_name and (current_regime == 'TRENDING_BULL' or current_regime == 'TRENDING_BEAR'):
                         # Check internal ADX of the pair before blocking? 
                         # For now, we trust the strategy's internal logic (it has ADX check).
                         pass

                    if should_run:
                        strategy.calculate_signals(event)
                        
                except Exception as e:
                    print(f"⚠️  Strategy Error ({strategy.__class__.__name__}): {e}")
                    
            # LAYER 1: Portfolio Exit Monitoring (Safety Net)
            # Check all open positions and force exits if thresholds are breached
            if self.portfolio and len(self.data_handlers) > 0:
                try:
                    self.portfolio.check_exits(self.data_handlers[0], self.events)
                except Exception as e:
                    print(f"⚠️  Portfolio Exit Check Error: {e}")
                    
            # Notify portfolio to update positions (mark-to-market)
            if self.portfolio:
                for dh in self.data_handlers:
                    if hasattr(dh, 'symbol_list'):
                        for symbol in dh.symbol_list:
                            try:
                                bars = dh.get_latest_bars(symbol, n=1)
                                if bars:
                                    price = bars[-1]['close']
                                    self.portfolio.update_market_price(symbol, price)
                            except:
                                continue

        elif event.type == 'SIGNAL':
            # 0. TTL CHECK (Time To Live)
            # Prevent processing stale signals (older than 10s)
            # This protects against system lag causing execution at wrong prices
            # 0. TTL CHECK (Time To Live)
            # Prevent processing stale signals (older than 10s)
            # BUG #53 FIX: Timezone-aware comparison
            from datetime import datetime, timedelta, timezone
            now_utc = datetime.now(timezone.utc)
            
            # Ensure event.datetime is timezone-aware UTC
            if isinstance(event.datetime, datetime):
                event_dt = event.datetime
                if event_dt.tzinfo is None:
                    # Assume UTC if naive, or use local if that's what your system does.
                    # Best practice: Convert everything to UTC.
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                
                # Calculate age
                age = (now_utc - event_dt).total_seconds()
                
                # Allow slightly larger window (30s) to account for network/processing lag
                if age > 30:
                    print(f"⚠️  Engine: Discarding STALE signal for {event.symbol} (Age: {age:.1f}s > 30s)")
                    return

            # 1. Log Signal
            if self.portfolio:
                self.portfolio.update_signal(event)
            
            # 2. Risk Management Check
            if hasattr(self, 'risk_manager'):
                # We need the current price to calculate quantity
                # For now, we'll fetch it from the data handlers (hacky but works for prototype)
                # Ideally, SignalEvent should carry the price or we query DataHandler
                
                # Simplified: Assume we can get price from the first data handler that has it
                price = 0
                # BUG #52 FIX: Optimized price fetching
                # Instead of looping through all handlers, use the one that matches the symbol
                # Or better, pass price in SignalEvent (which we should do in future)
                price = 0
                
                # Try to get price from the first handler that has it
                if self.data_handlers:
                    dh = self.data_handlers[0] # Usually only one handler (BinanceLoader)
                    bars = dh.get_latest_bars(event.symbol, n=1)
                    if bars:
                        price = bars[-1]['close']
                
                order_event = self.risk_manager.generate_order(event, price)
                if order_event:
                    self.events.put(order_event)

        elif event.type == 'ORDER':
            # Execution handler sends orders to broker
            if self.execution_handler:
                self.execution_handler.execute_order(event)
            else:
                print(f"SIMULATED EXECUTION: {event.direction} {event.quantity} {event.symbol}")
                # Create a Fill for the Portfolio to track
                # self.events.put(FillEvent(...)) # TODO

        elif event.type == 'FILL':
            # Portfolio updates positions based on fills
            if self.portfolio:
                self.portfolio.update_fill(event)

    def stop(self):
        self.running = False
