"""
Event-Driven Trading Engine - Optimized Version
Coordinates data, strategies, risk, and execution with enhanced validation and resource management.
"""

from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
import queue
import time
import threading
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from config import Config
from utils.debug_tracer import trace_execution
from utils.logger import logger
from core.world_awareness import world_awareness
from utils.latency_monitor import latency_monitor
from utils.system_monitor import system_monitor

class BoundedQueue:
    """
    High-Performance Lock-Free(ish) Queue.
    Wraps standard queue.Queue but optimized for discarding old events.
    """
    def __init__(self, maxsize=1000):
        self._q = queue.Queue(maxsize=maxsize)
    
    def put(self, item, block=True, timeout=None):
        try:
            self._q.put(item, block=False)
        except queue.Full:
            try:
                self._q.get_nowait() # Discard oldest
            except queue.Empty:
                pass
            
            try:
                self._q.put(item, block=False)
            except queue.Full:
                pass # Drop if still full (extreme load)

    def get(self, block=True, timeout=None):
        return self._q.get(block=block, timeout=timeout)

    def empty(self):
        return self._q.empty()

class Engine:
    """
    Event-Driven Trading Engine - EXTREME OPTIMIZATION EDITION
    - Zero global locks (Lock-free orchestration)
    - Direct payload processing (No lookups)
    - Burst-capable Event Loop
    """
    def __init__(self, events_queue=None):
        self.events = events_queue if events_queue else BoundedQueue(maxsize=1000)
        self.data_handlers = []
        self.strategies = []
        self.execution_handler = None
        self.portfolio = None
        self.risk_manager = None
        self.order_manager = None
        self.running = True
        
        # Strategy Coordination
        self._strategy_cooldowns = {}
        # REMOVED: self._event_lock = threading.RLock() -> Global lock killed latency
        
        # Metrics (optimized int counters)
        self.metrics = {
            'processed_events': 0,
            'discarded_events': 0,
            'strategy_executions': 0,
            'errors': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0
        }

    # ... [Registration methods unchanged] ...
    def register_data_handler(self, handler: Any) -> None: 
        self.data_handlers.append(handler)
        
    def register_strategy(self, strategy: Any) -> None: 
        self.strategies.append(strategy)
        
    def register_portfolio(self, portfolio: Any) -> None: 
        self.portfolio = portfolio
        
    def register_execution_handler(self, handler: Any) -> None: 
        self.execution_handler = handler
        
    def register_risk_manager(self, manager: Any) -> None: 
        self.risk_manager = manager

    def register_order_manager(self, manager: Any) -> None:
        self.order_manager = manager

    def unregister_strategy(self, symbol: str) -> None:
        """Removes all strategies associated with a symbol."""
        to_remove = [s for s in self.strategies if getattr(s, 'symbol', None) == symbol]
        for s in to_remove:
            try:
                s.stop()
                self.strategies.remove(s)
            except Exception as e:
                logger.error(f"Error unregistering strategy for {symbol}: {e}")
        
        if to_remove:
            logger.info(f"♻️ Engine: Unregistered {len(to_remove)} strategies for {symbol}")

    def run(self):
        """Main event loop - Optimized for Speed"""
        logger.info(f"🚀 Engine started. Active Strategies: {len(self.strategies)}")
        
        # Localize variables for speed
        get_event = self.events.get
        process = self.process_event
        perf_counter = time.perf_counter
        
        while self.running:
            try:
                # 0.1s timeout is fine for checking 'running' flag
                # For HFT, we might want purely blocking but then we can't stop easily.
                # 0.1s is acceptable.
                event = get_event(timeout=0.1)
                
                # METRICS: Only sample 1% of calls or if DEBUG is on to save CPU
                # t0 = perf_counter()
                process(event)
                # dt = (perf_counter() - t0) * 1000
                
                # Simple metric tracking without overhead
                self.metrics['processed_events'] += 1
                
            except queue.Empty:
                 # Phase 9: Active Order Management
                if self.order_manager:
                    self.order_manager.check_active_orders()

                # Phase 15: System Monitoring
                try:
                    system_monitor.check_health()
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    
                continue
            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
                self.metrics['errors'] += 1

    # REMOVED @trace_execution (Too slow for hot path)
    def process_event(self, event: Union[MarketEvent, SignalEvent, OrderEvent, FillEvent]) -> None:
        """Route event w/o Global Lock"""
        try:
            etype = event.type
            if etype == 'MARKET':
                self._process_market_event(event) # type: ignore
            elif etype == 'SIGNAL':
                self._process_signal_event(event) # type: ignore
            elif etype == 'ORDER':
                self._process_order_event(event) # type: ignore
            elif etype == 'FILL':
                self._process_fill_event(event) # type: ignore
            elif etype == 'AUDIT':
                pass 
            else:
                pass
        except Exception as e:
            logger.error(f"Event Logic Error {event.type}: {e}", exc_info=False)
            self.metrics['errors'] += 1

    def _process_market_event(self, event: MarketEvent) -> None:
        """Process MARKET event: Update portfolio prices and run strategies"""
        
        # 1. Efficient Portfolio Update (Active symbols only)
        # OPTIMIZATION: Use payload if available, else fallback
        if event.symbol and event.close_price and self.portfolio:
             self.portfolio.update_market_price(event.symbol, event.close_price)
             # Also execute exit check immediately for this symbol
             self._check_exits_fast(event)
        else:
             # Legacy/Mass Update fallback (slower)
             self._update_portfolio_prices()
        
        # 2. Market Regime Detection (Only if significant time passed or high volatility)
        current_regime = 'UNKNOWN' # Optimization: Delay regime calc until strat needs it
        
        # 3. Global Context (Lazy Load)
        context = None 
        
        # 4. Strategy Orchestration
        for strategy in self.strategies:
            # Quick filter before heavy lifting
            if hasattr(strategy, 'symbol') and strategy.symbol != event.symbol:
                continue

            # Lazy load Regime & Context only if we found a matching strategy
            if current_regime == 'UNKNOWN':
                 current_regime = self._get_current_market_regime()
            
            if self._should_strategy_run(strategy, event, current_regime):
                try:
                    if context is None:
                         context = world_awareness.get_market_context()
                    
                    strategy.market_context = context
                    strategy.calculate_signals(event)
                    self.metrics['strategy_executions'] += 1
                except Exception as e:
                    logger.error(f"Strategy Error ({strategy.__class__.__name__}): {e}")

    def _check_exits_fast(self, event):
        """Optimized exit checker for single symbol"""
        if self.portfolio and event.symbol in self.portfolio.positions:
             # We can't easily call 'check_exits' from portfolio because it iterates ALL.
             # But Portfolio.check_exits iterates all.
             # For now, calling the full check is safer but we should make it targeted.
             # Let's trust Portfolio's loop for now (it's in memory, fast enough for <10 positions).
             if self.data_handlers:
                  self.portfolio.check_exits(self.data_handlers[0], self.events)

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
                # Track Signal-to-Order latency
                dt_ms = (time.time_ns() - event.timestamp_ns) / 1_000_000
                latency_monitor.track('signal_to_order', dt_ms)
                self.events.put(order_event)

    def _process_order_event(self, event):
        """Process ORDER event: Execute via Execution Handler"""
        if self.execution_handler:
            self.execution_handler.execute_order(event)
        else:
            logger.warning("No Execution Handler registered. Order ignored.")

    def _process_fill_event(self, event):
        """Process FILL event: Update Portfolio and Order Manager"""
        # Track E2E latency if timestamp available
        dt_ms = (time.time_ns() - event.timestamp_ns) / 1_000_000
        latency_monitor.track('e2e_signal_to_fill', dt_ms)
        
        if self.portfolio:
            self.portfolio.update_fill(event)
        
        # Phase 9: Notify OrderManager
        if self.order_manager and hasattr(event, 'order_id') and event.order_id:
            self.order_manager.remove_order(event.order_id)

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
        
        # 0. Symbol Matching (Rule 4.2)
        # PROFESSOR METHOD: No procesar eventos de otros símbolos para reducir latencia.
        if hasattr(strategy, 'symbol') and strategy.symbol != event.symbol:
            return False

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
        for strategy in self.strategies:
            try:
                strategy.stop()
            except Exception as e:
                logger.error(f"Error stopping strategy {getattr(strategy, 'symbol', 'Unknown')}: {e}")
