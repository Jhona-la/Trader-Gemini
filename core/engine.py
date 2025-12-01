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
                    # ORCHESTRATION LOGIC (STRICT GATING)
                    strat_name = strategy.__class__.__name__
                    should_run = True
                    
                    if current_regime == 'TRENDING_BULL' or current_regime == 'TRENDING_BEAR':
                        # TRENDING MARKETS:
                        # ✅ ALLOW: MLStrategy (Trend Following), PatternStrategy (Continuation)
                        # ❌ BLOCK: StatisticalStrategy (Mean Reversion), TechnicalStrategy (RSI Reversion)
                        if 'Statistical' in strat_name or 'Technical' in strat_name:
                            should_run = False
                            # print(f"  🛑 Blocking {strat_name} in {current_regime} (Trend Mode)")
                    
                    elif current_regime == 'RANGING':
                        # SIDEWAYS MARKETS:
                        # ✅ ALLOW: StatisticalStrategy (Pairs), TechnicalStrategy (RSI Reversion)
                        # ❌ BLOCK: MLStrategy (Trend Following - Whipsaw Risk)
                        if 'MLStrategy' in strat_name:
                            should_run = False
                            # print(f"  🛑 Blocking {strat_name} in {current_regime} (Range Mode)")
                            
                    elif current_regime == 'CHOPPY':
                        # UNCERTAIN MARKETS:
                        # ✅ ALLOW: TechnicalStrategy (Scalping), PatternStrategy (Short-term)
                        # ❌ BLOCK: MLStrategy, StatisticalStrategy (Too risky)
                        if 'MLStrategy' in strat_name or 'Statistical' in strat_name:
                            should_run = False
                            
                    if should_run:
                        strategy.calculate_signals(event)
                        
                except Exception as e:
                    print(f"⚠️  Strategy Error ({strategy.__class__.__name__}): {e}")
                    # import traceback
                    # traceback.print_exc()
            # Notify portfolio to update positions (mark-to-market)
            if self.portfolio:
                # MarketEvent is generic, so we update all symbols
                # We iterate over data handlers to get symbols
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
                for dh in self.data_handlers:
                    try:
                        bars = dh.get_latest_bars(event.symbol, n=1)
                        if bars:
                            price = bars[-1]['close']
                            break
                    except:
                        continue
                
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
