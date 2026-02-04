"""
Event-Driven Trading Engine - Optimized Version
Coordinates data, strategies, risk, and execution with enhanced validation and resource management.
"""

from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
import queue
import time
import threading
from typing import Optional
from datetime import datetime, timezone
from config import Config
from utils.debug_tracer import trace_execution
from utils.logger import logger

class BoundedQueue(queue.Queue):
    """
    Queue with maxsize and aggressive discard policy for oldest events.
    Prevents memory leaks during high frequency data bursts.
    """
    def __init__(self, maxsize=1000):
        super().__init__(maxsize=maxsize)
    
    def put(self, item, block=True, timeout=None):
        try:
            # Try to put non-blocking first
            super().put(item, block=False)
        except queue.Full:
            # If full, discard oldest item to make space
            try:
                self.get_nowait()
            except queue.Empty:
                pass
            
            # Try again (should succeed now, but handle race condition)
            try:
                super().put(item, block=False)
            except queue.Full:
                logger.warning("Event queue full even after discard - dropping event")

class Engine:
    """
    Event-Driven Trading Engine with:
    - Real Market Regime Detection
    - Smart Strategy Coordination
    - Efficient Portfolio Updates
    - Price Validation
    """
    def __init__(self, events_queue=None):
        self.events = events_queue if events_queue else BoundedQueue(maxsize=500)
        self.data_handlers = []
        self.strategies = []
        self.execution_handler = None
        self.portfolio = None
        self.risk_manager = None
        self.running = True
        
        # Strategy Coordination
        self._strategy_cooldowns = {}
        self._cooldown_duration = 60  # 1 minute between signals for same strategy/symbol
        self._event_lock = threading.RLock()
        
        # Metrics
        self.metrics = {
            'processed_events': 0,
            'discarded_events': 0,
            'strategy_executions': 0,
            'errors': 0
        }

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
        """Main event loop"""
        logger.info(f"🚀 Engine started. Active Strategies: {len(self.strategies)}")
        while self.running:
            try:
                # Non-blocking get with timeout to allow clean exit check
                event = self.events.get(timeout=0.1)
                self.process_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
                self.metrics['errors'] += 1

    @trace_execution
    def process_event(self, event):
        """Route event to appropriate handlers with validation"""
        self.metrics['processed_events'] += 1
        
        with self._event_lock:
            try:
                if event.type == 'MARKET':
                    self._process_market_event(event)
                elif event.type == 'SIGNAL':
                    self._process_signal_event(event)
                elif event.type == 'ORDER':
                    self._process_order_event(event)
                elif event.type == 'FILL':
                    self._process_fill_event(event)
                elif event.type == 'AUDIT':
                    pass # Audit events are for logging only
                else:
                    logger.debug(f"Unknown event type: {event.type}")
            except Exception as e:
                logger.error(f"Error processing event {event.type}: {e}", exc_info=True)
                self.metrics['errors'] += 1

    def _process_market_event(self, event):
        """Process MARKET event: Update portfolio prices and run strategies"""
        
        # 1. Efficient Portfolio Update (Active symbols only)
        self._update_portfolio_prices()
        
        # 2. Market Regime Detection
        current_regime = self._get_current_market_regime()
        
        # 3. Strategy Orchestration
        for strategy in self.strategies:
            if self._should_strategy_run(strategy, event, current_regime):
                try:
                    strategy.calculate_signals(event)
                    self.metrics['strategy_executions'] += 1
                except Exception as e:
                    logger.error(f"Strategy Error ({strategy.__class__.__name__}): {e}")

        # 4. Portfolio Exit Safety Net
        if self.portfolio and self.data_handlers:
            try:
                self.portfolio.check_exits(self.data_handlers[0], self.events)
            except Exception as e:
                logger.error(f"Portfolio Exit Check Error: {e}")

    def _process_signal_event(self, event):
        """Process SIGNAL event: Validate and route to Risk Manager"""
        
        # 0. TTL Check
        if not self._validate_signal_ttl(event):
            self.metrics['discarded_events'] += 1
            return
            
        # 1. Price Validation
        current_price = self._get_validated_price(event.symbol)
        if not current_price:
            logger.warning(f"Discarding signal for {event.symbol}: Unable to validate price")
            self.metrics['discarded_events'] += 1
            return
            
        # 2. Log Signal (if portfolio is available)
        if self.portfolio:
             self.portfolio.update_signal(event)

        # 3. Risk Management
        if self.risk_manager:
            order_event = self.risk_manager.generate_order(event, current_price)
            if order_event:
                self.events.put(order_event)

    def _process_order_event(self, event):
        """Process ORDER event: Execute via Execution Handler"""
        if self.execution_handler:
            self.execution_handler.execute_order(event)
        else:
            logger.warning("No Execution Handler registered. Order ignored.")

    def _process_fill_event(self, event):
        """Process FILL event: Update Portfolio"""
        if self.portfolio:
            self.portfolio.update_fill(event)

    # ==================================================================
    # HELPER METHODS
    # ==================================================================

    def _get_current_market_regime(self) -> str:
        """Determines current market regime (TRENDING_BULL, TRENDING_BEAR, RANGING, HIGH_VOLATILITY)"""
        # 1. Trust Risk Manager first (centralized analysis)
        if self.risk_manager and hasattr(self.risk_manager, 'current_regime'):
             return self.risk_manager.current_regime
        
        # 2. Fallback: Simple heuristic using BTC proxy
        # This prevents 'RANGING' deadlock if risk manager isn't updating
        try:
            if self.data_handlers:
                dh = self.data_handlers[0]
                bars = dh.get_latest_bars('BTC/USDT', n=50)
                if bars and len(bars) >= 20:
                     # TODO: Implement local regime fallback logic here or utility
                     pass
        except Exception:
            pass
            
        return 'UNKNOWN'

    def _should_strategy_run(self, strategy, event, regime: str) -> bool:
        """
        Coordination Logic:
        - Prevents conflicting strategies
        - Enforces regime compatibility
        """
        strat_name = strategy.__class__.__name__
        
        # 1. Regime Compatibility
        if 'Statistical' in strat_name:
            # Mean reversion is dangerous in strong trends
            if 'TRENDING' in regime:
                 return False
                 
        if 'ML' in strat_name:
            # ML typically trained for trends
            if regime == 'CHOPPY':
                # Optional: reduce frequency or block
                pass
                
        # 2. Existing Position Check (Optional: prevent fighting own position)
        if self.portfolio and hasattr(self.portfolio, 'positions'):
            pos = self.portfolio.positions.get(event.symbol)
            if pos and pos['quantity'] != 0:
                # If we have a position, only allow strategies that manage exits or pyramids
                # For simplicity in this engine, we let them run but RiskManager filters adds
                pass

        return True

    def _update_portfolio_prices(self):
        """Update market prices ONLY for symbols with open positions"""
        if not self.portfolio or not self.data_handlers:
            return
            
        active_symbols = [
            sym for sym, pos in self.portfolio.positions.items()
            if pos['quantity'] != 0
        ]
        
        if not active_symbols:
            return

        dh = self.data_handlers[0]
        for symbol in active_symbols:
            try:
                bars = dh.get_latest_bars(symbol, n=1)
                if bars:
                    self.portfolio.update_market_price(symbol, bars[-1]['close'])
            except Exception:
                continue

    def _get_validated_price(self, symbol: str) -> Optional[float]:
        """
        Get and validate current price.
        Checks for:
        - Freshness (availability)
        - Non-zero validity
        - Anomalous jumps (optional simple check)
        """
        if not self.data_handlers:
            return None
            
        try:
            dh = self.data_handlers[0]
            # Fetch recent bars to validate
            bars = dh.get_latest_bars(symbol, n=3)
            
            if not bars:
                return None
                
            current_price = bars[-1]['close']
            
            if current_price <= 0:
                return None
                
            # Basic anomaly check (spike detection)
            if len(bars) >= 2:
                prev_price = bars[-2]['close']
                if prev_price > 0:
                    pct_change = abs(current_price - prev_price) / prev_price
                    if pct_change > 0.15: # >15% jump in one timeframe is suspicious
                        logger.warning(f"Price anomaly detected for {symbol}: {pct_change*100:.1f}% jump")
                        return None
                        
            return current_price
            
        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            return None

    def _validate_signal_ttl(self, event) -> bool:
        """Check if signal is too old to process"""
        now = datetime.now(timezone.utc)
        age = (now - event.datetime).total_seconds()
        
        if age > Config.MAX_SIGNAL_AGE:
            if age > 5.0: # Log only significant delays
                 logger.warning(f"Discarding STALE signal {event.symbol} (Age: {age:.2f}s)")
            return False
            
        return True

    def stop(self):
        self.running = False
