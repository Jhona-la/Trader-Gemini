"""
Event-Driven Trading Engine - Optimized Version
Coordinates data, strategies, risk, and execution with enhanced validation and resource management.
"""

from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
import asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import collections
from typing import Optional, Any
from config import Config
from utils.logger import logger
from utils.os_tuner import OSTuner # Protocol Nadir-Soberano
from utils.time_sync import TimeSynchronizer # Phase 26: Stochastic Purity
from core.system_monitor import SystemMonitor # Phase 27: Disaster Resilience
from config import Config
from utils.logger import logger
from utils.latency_monitor import latency_monitor
from core.gc_tuner import GCTuner
from core.forensics import ForensicRecorder # Phase 20: Forensic Logging

try:
    import psutil
except ImportError:
    psutil = None

class AsyncBoundedQueue:
    """
    HFT Ring Buffer Queue (Phase 2: Core Refactor).
    - Uses collections.deque(maxlen) for O(1) auto-drop of oldest events.
    - Non-blocking PUT (Always succeeds). Sync-safe for strategy calls.
    - Async GET (Waits for data) with double-check to prevent lost wakeups.
    
    [SS-001 FIX] Double-check pattern prevents TOCTOU race in get().
    [SS-002 FIX] put() is now sync (non-async) so strategies can call
    it without `await`. deque.append() is thread-safe, and _event.set()
    is safe to call from within the same event loop thread.
    """
    def __init__(self, maxsize=1000):
        self._deque = collections.deque(maxlen=maxsize)
        self._event = asyncio.Event()
    
    def put(self, item):
        """Put item into queue. Never blocks. Sync-safe for strategy calls."""
        self._deque.append(item)
        self._event.set()

    async def get(self):
        """Wait for and get next item. Double-check prevents lost wakeups."""
        while True:
            # Fast path: data available
            if self._deque:
                return self._deque.popleft()
            
            # Slow path: wait for signal with double-check
            self._event.clear()
            
            # [SS-001] Re-check AFTER clear to prevent race:
            # If put() fired set() between our first check and clear(),
            # the data is in _deque but event was just cleared.
            if self._deque:
                return self._deque.popleft()
            
            await self._event.wait()

    def empty(self):
        return not self._deque

    def task_done(self):
        pass # Not tracked for speed

from utils.metrics_exporter import metrics

class Engine:
    """
    Event-Driven Trading Engine - EXTREME OPTIMIZATION EDITION
    - Zero global locks (Lock-free orchestration)
    - Direct payload processing (No lookups)
    - Burst-capable Event Loop
    """
    def __init__(self, events_queue: Optional[Any] = None):
        self.events = events_queue if events_queue else AsyncBoundedQueue(maxsize=5000) # Increased buffer for burst
        self.data_handlers = []
        self.strategies = []
        self.execution_handler = None
        self.portfolio = None
        self.risk_manager = None
        self.order_manager = None
        self.running = True
        
        # Strategy Coordination
        self._strategy_cooldowns = {}
        
        # Metrics (optimized int counters)
        self.metrics = {
            'processed_events': 0,
            'discarded_events': 0,
            'strategy_executions': 0,
            'errors': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'burst_events': 0,  # Phase OMNI: Burst-mode drain counter
        }
        
        # 🕵️ Phase 20: Forensic Recorder
        self.forensics = ForensicRecorder(self)
        
        # 🌑 PHASE 24: LAYER 0 OPTIMIZATION (Protocol Nadir-Soberano)
        OSTuner.optimize()
        
        # 🧬 PHASE 26: Time Synchronization Integration
        TimeSynchronizer.sync()
        
        # 🏥 PHASE 27: System Monitor
        self.system_monitor = SystemMonitor()

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
        # 🛡️ Phase 20: Link Forensics to Kill Switch
        if self.risk_manager and hasattr(self.risk_manager, 'kill_switch'):
             # Define callback that captures snapshot
             def forensic_dump(reason):
                 if hasattr(self, 'forensics'):
                     self.forensics.capture_snapshot(trigger_reason=f"KILL_SWITCH: {reason}")
             
             self.risk_manager.kill_switch.set_forensic_callback(forensic_dump)

    def register_order_manager(self, manager: Any) -> None:
        self.order_manager = manager

    def unregister_strategy(self, symbol: str) -> None:
        """Removes all strategies associated with a symbol."""
        to_remove = [s for s in self.strategies if getattr(s, 'symbol', None) == symbol]
        for s in to_remove:
            try:
                if hasattr(s, 'stop'):
                    s.stop()
                self.strategies.remove(s)
            except Exception as e:
                logger.error(f"Error unregistering strategy for {symbol}: {e}")
        
        if to_remove:
            logger.info(f"♻️ Engine: Unregistered {len(to_remove)} strategies for {symbol}")

    async def start(self):
        """Main event loop - 100% AsyncIO-Driven"""
        # AEGIS-ULTRA: Core Pinning (Phase 5)
        if Config.Aegis.CORE_PINNING and psutil:
            try:
                p = psutil.Process()
                # Pin to physical cores only (Ryzen 7 5700U: 0, 2, 4, 6, 8, 10, 12, 14)
                # We select the first 4 physical cores for the main Engine loop
                physical_cores = [0, 2, 4, 6] 
                p.cpu_affinity(physical_cores)
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info(f"🛡️ AEGIS-ULTRA: Engine Pinned to Cores {physical_cores} | Priority: HIGH")
            except Exception as e:
                logger.warning(f"⚠️ Failed to set CPU Affinity: {e}")
        
        # ✅ PHASE IV: COLD BOOT (Operational Continuity)
        # Verify if we need to recover state from Exchange
        if self.executor and self.portfolio:
             try:
                 logger.info("🔌 [COLD BOOT] Initiating State Recovery Protocol...")
                 self.executor.sync_portfolio_state(self.portfolio)
             except Exception as e:
                 logger.critical(f"❌ [COLD BOOT] FAILED: {e} - Proceeding with local state.")
        
        logger.info(f"🚀 Engine started. Active Strategies: {len(self.strategies)}")
        
        while self.running:
            try:
                # 🏥 PHASE 27: Graceful Degradation Check
                if not self.system_monitor.check_health():
                    await asyncio.sleep(0.1) # Cool down

                # 1. Get first event (Wait max 1s to allow maintenance)
                try:
                    event = await asyncio.wait_for(self.events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Idle Cycle: Smart GC (Phase 15)
                    regime = self._get_current_market_regime()
                    if 'VOLATILE' not in regime and 'TRENDING' not in regime:
                        GCTuner.check_maintenance()
                    continue
                
                # ⚡ PHASE OMNI: BURST-MODE EVENT DRAIN
                # Process up to 32 events per yield cycle to reduce asyncio overhead.
                # QUÉ: Drain multiple events in a single GC-free critical section.
                # POR QUÉ: Cada await cede control al event loop (~15μs overhead).
                # PARA QUÉ: Reducir latencia total bajo carga alta (bursts de mercado).
                # CÓMO: Recoger hasta 32 eventos de la deque sin await entre ellos.
                burst_batch = [event]
                _BURST_MAX = 32
                while len(burst_batch) < _BURST_MAX and not self.events.empty():
                    try:
                        burst_batch.append(self.events._deque.popleft())
                    except IndexError:
                        break
                
                # 2. Critical Section (GC Disabled) — process entire burst
                start_loop = time.perf_counter()
                with GCTuner.critical_section():
                    for evt in burst_batch:
                        await self.process_event(evt)

                # [DF-A2] Jitter Detection: Measure processing time
                end_loop = time.perf_counter()
                batch_size = len(burst_batch)
                loop_duration = (end_loop - start_loop) * 1_000_000 # microseconds
                per_event_us = loop_duration / batch_size
                
                if per_event_us > 500: # 500μs per-event threshold
                    logger.warning(f"⚠️ [JITTER] Burst({batch_size}) Avg: {per_event_us:.0f}μs/evt (Total: {loop_duration:.0f}μs)")
                    latency_monitor.track('engine_jitter_warning', loop_duration / 1000)
                
                # Update rolling avg latency
                for _ in burst_batch:
                    current_avg = self.metrics['avg_latency_ms']
                    processed = self.metrics['processed_events']
                    self.metrics['avg_latency_ms'] = (current_avg * processed + (per_event_us/1000)) / (processed + 1)
                    self.metrics['processed_events'] += 1
                
                if (loop_duration/1000) > self.metrics['max_latency_ms']:
                    self.metrics['max_latency_ms'] = loop_duration / 1000
                
                # Track burst size for telemetry
                if batch_size > 1:
                    self.metrics['burst_events'] += batch_size
                
                # Mark as done
                for _ in burst_batch:
                    self.events.task_done()
                
                # ✅ PHASE 18: RYZEN 7 SNIPER (Dynamic Orchestration)
                if self.metrics['processed_events'] % 100 == 0:
                     self._optimize_ryzen_resources()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
                self.metrics['errors'] += 1

    def _optimize_ryzen_resources(self):
        """
        [PHASE 18] Dynamic Core Pinning & Thermal Throttling.
        """
        if not psutil: return
        try:
            # 1. Thermal Check
            # Windows psutil doesn't always support temperatures, but we try.
            # If unavailable, we skip.
            throttle = 0.0
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Find max CPU temp
                    max_temp = 0
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current > max_temp: max_temp = entry.current
                    
                    if max_temp > 80.0:
                        logger.warning(f"🔥 [THERMAL] CPU at {max_temp}°C. Throttling active.")
                        throttle = 0.1 # 100ms penalty
                        time.sleep(throttle) # Force cool down
            
            # 2. Dynamic Affinity (Load Balancer)
            cpu_pct = psutil.cpu_percent()
            p = psutil.Process()
            
            if cpu_pct > 80.0:
                # High Load: Unleash all Cores (Ryzen 7 5700U has 8 cores / 16 threads)
                # We use all logical processors
                current_affinity = p.cpu_affinity()
                if len(current_affinity) < 16:
                    p.cpu_affinity(list(range(16)))
                    logger.info(f"⚡ [SNIPER] High Load ({cpu_pct}%). Expanding to ALL 16 Threads.")
            elif cpu_pct < 20.0:
                # Low Load: Conserve Power (Eco Mode)
                # Pin to first 4 physical cores (0, 2, 4, 6)
                current_affinity = p.cpu_affinity()
                target = [0, 2, 4, 6]
                if current_affinity != target:
                    p.cpu_affinity(target)
                    logger.info(f"🍃 [SNIPER] Low Load ({cpu_pct}%). Eco Mode Active (4 Cores).")
                    
        except Exception:
            pass # Fail silently on permission/OS errors

    async def process_event(self, event: Union[MarketEvent, SignalEvent, OrderEvent, FillEvent]) -> None:
        """Route event asynchronously"""
        try:
            metrics.inc_event() # Phase 53: Metrics
            etype = event.type
            
            # AEGIS-ULTRA: LATENCY CIRCUIT BREAKER (Phase 16)
            # Check latency before processing Signals/Orders (Market data always processed)
            if etype in ['SIGNAL', 'ORDER'] and self.data_handlers:
                dh = self.data_handlers[0]
                if hasattr(dh, 'get_latency_metrics'):
                    avg_ping, max_ping = dh.get_latency_metrics()
                    if avg_ping > 150.0:
                        if etype == 'SIGNAL':
                            logger.warning(f"🛑 [CIRCUIT BREAKER] High Latency ({avg_ping:.1f}ms). Signal Dropped.")
                            self.metrics['discarded_events'] += 1
                            return
                        elif etype == 'ORDER':
                            # Optional: allow closing orders? For now block all to be safe against stale prices.
                            # Better: Allow CLOSE, Block ENTRY. But Engine doesn't know intent easily.
                            # Conservative: Block ALL new actions. Risk Manager handles emergency exits via direct API?
                            # For now, just log and block.
                            logger.warning(f"🛑 [CIRCUIT BREAKER] High Latency ({avg_ping:.1f}ms). Order Blocked.")
                            return

            if etype == 'MARKET':
                await self._process_market_event(event) # type: ignore
            elif etype == 'SIGNAL':
                await self._process_signal_event(event) # type: ignore
            elif etype == 'ORDER':
                await self._process_order_event(event) # type: ignore
            elif etype == 'FILL':
                await self._process_fill_event(event) # type: ignore
            elif etype == 'AUDIT':
                pass 
            else:
                pass
        except Exception as e:
            logger.error(f"Event Logic Error {event.type}: {e}", exc_info=False)
            self.metrics['errors'] += 1

    async def _process_market_event(self, event: MarketEvent) -> None:
        """Process MARKET event asynchronously"""
        
        # 1. Efficient Portfolio Update (Active symbols only)
        if event.symbol and event.close_price and self.portfolio:
             self.portfolio.update_market_price(event.symbol, event.close_price)
             await self._check_exits_fast(event)
        else:
             self._update_portfolio_prices()
        
        # 2. Market Regime Detection
        current_regime = 'UNKNOWN' # Optimization: Delay regime calc until strat needs it
        
        # 3. Global Context
        context = None 
        
        # 4. Strategy Orchestration
        for strategy in self.strategies:
            if hasattr(strategy, 'symbol') and strategy.symbol != event.symbol:
                continue
 
            if current_regime == 'UNKNOWN':
                 current_regime = self._get_current_market_regime()
            
            if self._should_strategy_run(strategy, event, current_regime):
                try:
                    if context is None:
                         context = world_awareness.get_market_context()
                    
                    strategy.market_context = context
                    
                    # Handle both sync and async calculate_signals
                    if asyncio.iscoroutinefunction(strategy.calculate_signals):
                        await strategy.calculate_signals(event)
                    else:
                        strategy.calculate_signals(event)
                        
                    self.metrics['strategy_executions'] += 1
                except Exception as e:
                    logger.error(f"Strategy Error ({strategy.__class__.__name__}): {e}")

    async def _check_exits_fast(self, event):
        """Optimized exit checker for single symbol - Async wrapper"""
        if self.portfolio and event.symbol in self.portfolio.positions:
             if self.data_handlers:
                  # Note: Portfolio.check_exits is currently sync
                  self.portfolio.check_exits(self.data_handlers[0], self.events)

    async def _process_signal_event(self, event):
        """Process SIGNAL event asynchronously"""
        
        if not self._validate_signal_ttl(event):
            self.metrics['discarded_events'] += 1
            return
            
        current_price = self._get_validated_price(event.symbol)
        if not current_price:
            logger.warning(f"Discarding signal for {event.symbol}: Unable to validate price")
            self.metrics['discarded_events'] += 1
            return
            
        if self.portfolio:
             self.portfolio.update_signal(event)

        if self.risk_manager:
            order_event = self.risk_manager.generate_order(event, current_price)
            if order_event:
                dt_ms = (time.time_ns() - event.timestamp_ns) / 1_000_000
                latency_monitor.track('signal_to_order', dt_ms)
                self.events.put(order_event)

    async def _process_order_event(self, event):
        """Process ORDER event asynchronously"""
        if self.execution_handler:
            if asyncio.iscoroutinefunction(self.execution_handler.execute_order):
                await self.execution_handler.execute_order(event)
            else:
                self.execution_handler.execute_order(event)
        else:
            logger.warning("No Execution Handler registered. Order ignored.")

    async def _process_fill_event(self, event):
        """Process FILL event asynchronously"""
        dt_ms = (time.time_ns() - event.timestamp_ns) / 1_000_000
        latency_monitor.track('e2e_signal_to_fill', dt_ms)
        
        if self.portfolio:
            result = self.portfolio.update_fill(event)
            if result is not None:
                # OMEGA MIND: Unpack result and outcome
                # Legacy support: if portfolio returns just pnl (float), handle it
                if isinstance(result, tuple):
                    pnl, trade_outcome = result
                else:
                    pnl = result
                    trade_outcome = 1.0 if pnl > 0 else 0.0 # Fallback for old strategies

                for strategy in self.strategies:
                    # Filtramos por símbolo para asegurar que actualizamos la instancia correcta
                    if getattr(strategy, 'symbol', None) == event.symbol and \
                       hasattr(strategy, 'update_recursive_weights'):
                        strategy.update_recursive_weights(trade_outcome)

        
        if self.order_manager and hasattr(event, 'order_id') and event.order_id:
            if asyncio.iscoroutinefunction(self.order_manager.remove_order):
                await self.order_manager.remove_order(event.order_id, event=event)
            else:
                self.order_manager.remove_order(event.order_id, event=event)

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

        # Phase 3: Hardware Optimization (GC Tuner)
        for strategy in self.strategies:
            try:
                strategy.stop()
            except Exception as e:
                logger.error(f"Error stopping strategy {getattr(strategy, 'symbol', 'Unknown')}: {e}")
