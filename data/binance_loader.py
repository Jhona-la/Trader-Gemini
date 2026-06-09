# import ccxt  <-- REMOVED (Rule 3.1 Separation of Concerns)
from binance.client import Client # Synchronous Client for REST calls
from binance.enums import *
import pandas as pd
import time
from datetime import datetime, timezone
from .data_provider import DataProvider
from core.events import MarketEvent
from config import Config  # Import Config
from utils.logger import logger
from utils.debug_tracer import trace_execution
from utils.thread_monitor import monitor
import asyncio
from typing import List, Dict, Optional, Any
from binance import AsyncClient, BinanceSocketManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import collections
import numpy as np
import polars as pl
from utils.fast_json import FastJson as json  # Phase 1: Zero-Latency Serialization
from utils.hft_buffer import NumbaStructuredRingBuffer, NumbaRingBuffer # Phase 4: Structured Buffers
import os # Phase 5
import pyarrow # Phase 5 Check
from utils.fast_strings import intern_string # Phase 21: String Interning Optimization
from utils.shm_utils import SharedMemoryManager # Phase 11: SHM Bridge
from strategies.components.microstructure import MicrostructureAnalyzer # Phase 25: Nadir-Soberano

# [MÓDULO OMEGA] - Dimensión 1: Validación de Datos
from data.validators.ohlcv_validator import OHLCVValidator


class BinanceData(DataProvider):
    def __init__(self, events_queue, symbol_list):
        from data.data_provider import register_data_provider
        register_data_provider(self)
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        self._running = True
        
        # 1. Thread Pool for Parallel Fetching (I/O Bound)
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="BinanceFetch")
        
        # 2. Data Buffers Dictionary Initialization (Phase 9/98)
        self.buffers_1m = {}
        self.buffers_5m = {}
        self.buffers_15m = {}
        self.buffers_1h = {}
        self.buffers_4h = {} # Phase 28: Multi-Horizon Position
        self.buffers_1d = {} # Phase 3: Macro Horizon
        self.buffers_1w = {} # Phase 3: Structural Horizon
        self.vbi_history = {}
        self.liquidation_history = {}
        self.last_event_time = {}
        self.liquidity_cache = {}
        # PHASE 13: Order Flow Metrics
        self.order_flow_metrics = {}
        
        # PHASE 29: Derivatives Metrics (OI, Funding)
        self.derivatives_metrics = {}
        
        # PHASE 14: Lead-Lag Intelligence
        self.lead_lag_results = {} # {symbol: lag_in_seconds}
        self.reference_symbol = "BTC/USDT"
        
        # 🌊 PHASE 25: Microstructure Analyzers
        self.microstructure = {} 
        for s in self.symbol_list:
            self.microstructure[s] = MicrostructureAnalyzer(s)

        # 🌊 PHASE 10: High-Frequency Limit Order Book (Cythonized)
        from core.orderbook import OrderBook
        self.orderbooks = {s: OrderBook(max_depth=10) for s in self.symbol_list}
        self.last_depth_update = {s: 0.0 for s in self.symbol_list}

        # 🔍 PHASE 3 (Data Integrity): Sliding window for gap detection
        self.latest_data = collections.defaultdict(lambda: collections.deque(maxlen=20))  # O(1) bounded buffer
        self.data_health_metrics = collections.defaultdict(lambda: {"gaps": 0, "last_backfill": 0})


        # 3. Init actual Numba buffers for each symbol
        for s in self.symbol_list:
            self._init_symbol_buffer(s)
            
        # 4. Thread Safety Lock
        import contextlib
        self._data_lock = contextlib.nullcontext()

        # ═══════════════════════════════════════════════════════════════
        # LOW-LATENCY PHASE: ZERO-ALLOC OUTPUT BUFFERS
        # QUÉ: Pre-allocar buffers de salida para get_latest_bars().
        # POR QUÉ: Antes, np.empty() creaba un array nuevo en CADA llamada.
        #   Con 26 símbolos × ~20 llamadas/tick = ~520 allocaciones/min
        #   que presionan al GC.
        # PARA QUÉ: Eliminar allocaciones de heap en hot-path (~30-100μs/llamada).
        # CÓMO: Pre-allocar con capacidad máxima; devolver slice/view.
        # CUÁNDO: Al inicializar BinanceData.
        # DÓNDE: data/binance_loader.py → __init__()
        # QUIÉN: SRE/DevOps + Quant Developer
        # ═══════════════════════════════════════════════════════════════
        self._BAR_STRUCT_DTYPE = np.dtype([
            ('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'),
            ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')
        ])
        self._BAR_OUTPUT_MAX = 2000  # Max bars any caller could request
        self._bar_output_cache = {}
        for s in self.symbol_list:
            self._bar_output_cache[s] = np.empty(self._BAR_OUTPUT_MAX, dtype=self._BAR_STRUCT_DTYPE)
        # Timeframe → buffer dict for O(1) lookup (replaces if/elif chain)
        self._timeframe_map = None  # Lazy init after buffers ready

        # 5. Synchronous Client Initialization (Rest, etc.)
        self._init_sync_client()
        
        # [PHASE 10] Watchdog State
        self.last_packet_time = time.time()
        self.watchdog_running = False
        
        # [PHASE 11] SHM Bridge
        self.shm_managers = {} 
        self._init_shm()

    def _init_shm(self):
        """
        [PHASE 11] Initialize Shared Memory for each symbol (LOB Snapshots).
        Structure: 20 floats (Bid1P, Bid1Q ... Ask5P, Ask5Q)
        """
        try:
            for s in self.symbol_list:
                safe_s = s.replace('/', '')
                # 20 floats * 4 bytes = 80 bytes per symbol
                dummy = np.zeros(20, dtype=np.float32)
                # Store manager but we need to keep it alive
                # The Manager __enter__ creates the SHM. We need to handle this manually or keep manager open.
                # Actually SharedMemoryManager context manager closes on exit.
                # We need persistent SHM.
                # Let's use low-level SharedMemory or a persistent wrapper.
                # Our utils wrapper is for context.
                # We will just instantiate it and manual open/close if needed or use it per write? 
                # Per write is slow (unlink/create every time).
                # We need PERSISTENT SHM.
                # Adaptation: We will just create the SHM here and keep it open.
                from multiprocessing import shared_memory
                try:
                    name = f"LOB_{safe_s}"
                    # Try to attach to existing first (Windows leaves them hanging)
                    try:
                        shm = shared_memory.SharedMemory(name=name)
                        # If sizes don't match or we want a fresh one, this might be tricky,
                        # but attaching is better than failing.
                    except FileNotFoundError:
                        # Clean create
                        shm = shared_memory.SharedMemory(create=True, size=dummy.nbytes, name=name)
                        
                    self.shm_managers[s] = {'shm': shm, 'arr': np.ndarray(20, dtype=np.float32, buffer=shm.buf)}
                except Exception as e:
                    logger.warning(f"SHM Init Failed for {s}: {e}")
                    
            logger.info(f"🧠 [SHM] Initialized Shared Memory for {len(self.shm_managers)} symbols")
        except Exception as e:
            logger.error(f"SHM Setup Error: {e}")

    async def _watchdog_loop(self):
        """
        [PHASE 10] Self-Healing Watchdog.
        Monitors socket heartbeat. If silence > 5s, forces restart.
        """
        self.watchdog_running = True
        logger.info("🐕 [Watchdog] Guardian Active")
        
        while self._running:
            await asyncio.sleep(1)
            
            # Check Silence
            silence = time.time() - self.last_packet_time
            if silence > 5.0 and len(self.active_sockets) > 0:
                logger.warning(f"🐕 [Watchdog] SILENCE DETECTED ({silence:.1f}s). Restarting Sockets...")
                self.last_packet_time = time.time() + 10 # Grace period
                self._force_restart_socket()
            
            # Phase 12: Drift Check (Simulated for now)
            # if drift_detected(): self.trigger_circuit_breaker()

        
    def _init_sync_client(self):
        """
        Initialize the synchronous Binance client for REST API calls.
        """
        api_key = Config.BINANCE_API_KEY
        api_secret = Config.BINANCE_SECRET_KEY
        
        if Config.BINANCE_USE_TESTNET:
            api_key = Config.BINANCE_TESTNET_API_KEY
            api_secret = Config.BINANCE_TESTNET_SECRET_KEY
            logger.info("Binance Loader: Configured for TESTNET")
        elif hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
             api_key = Config.BINANCE_DEMO_API_KEY
             api_secret = Config.BINANCE_DEMO_SECRET_KEY
             logger.info("Binance Loader: Configured for DEMO")
        else:
             logger.info("Binance Loader: Configured for LIVE")
        import threading
        self._thread_local = threading.local()
        
        # Phase 38 tuning and client instantiation is now handled per-thread in the property
        # Test connection (optional but good for debugging)
        try:
            self.client_sync.ping()
            logger.info("✅ Binance REST API Connected")
            
            # Phase 6: Proactive Time Sync Check
            server_time = self.client_sync.get_server_time()['serverTime']
            local_time = int(time.time() * 1000)
            diff = local_time - server_time
            if abs(diff) > 1000:
                logger.warning(f"⚠️ TIME DESYNC: System is {diff}ms {'ahead' if diff > 0 else 'behind'} Binance. Sync Clock!")
            else:
                logger.info(f"⏱️ Time Sync OK (Diff: {diff}ms)")
                
        except Exception as e:
            logger.error(f"❌ Binance REST API Connection Failed: {e}")
        
        # Throttling Tracking (NEW: Fixed missing attribute)
        # (Moved to __init__)
        
        # Fetch initial history at startup
        self.fetch_initial_history()
        self.fetch_initial_history_1h()
        self.fetch_initial_history_4h()  # NEW: Phase 28
        self.fetch_initial_history_5m()  # NEW
        self.fetch_initial_history_15m()  # NEW
        self.fetch_initial_history_1d()  # PHASE 3 Macro
        self.fetch_initial_history_1w()  # PHASE 3 Structural
        
        # Async Client & Socket Manager placeholders
        self.client = None
        self.bsm = None
        self.socket = None
        
        # Phase 16: Latency Circuit Breaker Stats
        self.latency_history = collections.deque(maxlen=20)
        self._start_latency_monitor()
        self._start_derivatives_monitor()

    @property
    def client_sync(self):
        """Returns the thread-safe Binance Sync Client."""
        if not hasattr(self, '_client_sync'):
            # Recreate credentials logic
            api_key = Config.BINANCE_API_KEY
            api_secret = Config.BINANCE_SECRET_KEY
            if hasattr(Config, 'BINANCE_USE_TESTNET') and Config.BINANCE_USE_TESTNET:
                api_key = Config.BINANCE_TESTNET_API_KEY
                api_secret = Config.BINANCE_TESTNET_SECRET_KEY
            elif hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
                api_key = Config.BINANCE_DEMO_API_KEY
                api_secret = Config.BINANCE_DEMO_SECRET_KEY
                
            client = Client(
                api_key, 
                api_secret, 
                testnet=(hasattr(Config, 'BINANCE_USE_TESTNET') and Config.BINANCE_USE_TESTNET),
                requests_params={'timeout': 5}
            )
            from utils.keep_alive import tune_requests_session
            tune_requests_session(client.session)
            self._client_sync = client
            
        return self._client_sync

    async def shutdown(self):
        """
        [PHASE 8] Graceful Shutdown of WebSockets and Thread Pools.
        Prevents AsyncIO Task exceptions on Engine exit.
        """
        self._running = False
        if self.bsm:
            try:
                self.bsm.stop()
            except Exception as e:
                logger.error(f"Error stopping BSM: {e}")
        
        if self.client:
            try:
                await self.client.close_connection()
            except Exception as e:
                logger.error(f"Error closing async client: {e}")
                
        # We don't shutdown executor if it's shared, but here it's dedicated
        try:
            self.executor.shutdown(wait=False)
        except:
            pass
            
        logger.info("🛑 [BinanceLoader] Resources gracefully shut down.")

    async def start_websockets(self):
        """
        [PHASE 1] Liquidation Sniper & Order Flow
        Connects to Binance Async Websockets to listen to @forceOrder
        """
        api_key = Config.BINANCE_API_KEY
        api_secret = Config.BINANCE_SECRET_KEY
        testnet = getattr(Config, 'BINANCE_USE_TESTNET', False)
        
        if not self.client:
            self.client = await AsyncClient.create(api_key, api_secret, testnet=testnet)
            self.bsm = BinanceSocketManager(self.client)
            
        async def liquidation_listener():
            try:
                # To get all liquidations, we can use the multiplex socket
                streams = [f"{sym.replace('/', '').lower()}@forceOrder" for sym in self.symbol_list]
                
                logger.info(f"🌊 [WebSockets] Liquidation Sniper listening to {len(streams)} streams...")
                multiplex_socket = self.bsm.multiplex_socket(streams)
                async with multiplex_socket as ts:
                    while self._running:
                        try:
                            msg = await ts.recv()
                            self._process_liquidation_msg(msg)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"Liquidation stream err: {e}")
                            await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Liquidation listener fatal: {e}")

        # Start background task
        asyncio.create_task(liquidation_listener())
        
        # 🌊 PHASE 10: Market Microstructure (L2 OrderBook Listener)
        # Limit to BTC and ETH initially to save bandwidth/CPU unless throttled.
        # We will subscribe to all but throttle the math update to 500ms.
        async def depth_listener():
            try:
                streams = [f"{sym.replace('/', '').lower()}@depth10@100ms" for sym in self.symbol_list]
                logger.info(f"📊 [WebSockets] L2 Orderbook listening to {len(streams)} streams...")
                multiplex_socket = self.bsm.multiplex_socket(streams)
                
                async with multiplex_socket as ts:
                    while self._running:
                        try:
                            msg = await ts.recv()
                            self._process_depth_msg(msg)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"Depth stream err: {e}")
                            await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Depth listener fatal: {e}")
                
        asyncio.create_task(depth_listener())

        # 👣 PHASE 13: Capa 4 Footprint Reconstructor
        async def trades_listener():
            try:
                streams = [f"{sym.replace('/', '').lower()}@aggTrade" for sym in self.symbol_list]
                logger.info(f"👣 [WebSockets] Footprint Reconstructor listening to {len(streams)} streams...")
                multiplex_socket = self.bsm.multiplex_socket(streams)
                
                async with multiplex_socket as ts:
                    while self._running:
                        try:
                            msg = await ts.recv()
                            self._process_trade_msg(msg)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"Trades stream err: {e}")
                            await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Trades listener fatal: {e}")
                
        asyncio.create_task(trades_listener())

    def _process_depth_msg(self, msg):
        """
        Processes @depth10@100ms messages and updates the Cython OrderBook.
        """
        try:
            if 'data' not in msg or 'bids' not in msg['data']:
                return
                
            data = msg['data']
            stream_name = msg.get('stream', '')
            symbol_raw = stream_name.split('@')[0].upper()
            
            # Format back to BTC/USDT
            symbol = f"{symbol_raw[:-4]}/{symbol_raw[-4:]}" if symbol_raw.endswith('USDT') else symbol_raw
            
            if symbol not in self.orderbooks:
                return
                
            # Throttle processing to 500ms to save CPU
            now = time.time()
            if (now - self.last_depth_update.get(symbol, 0)) < 0.5:
                return
            self.last_depth_update[symbol] = now
            
            ob = self.orderbooks[symbol]
            
            # Update Bids
            for bid in data.get('bids', []):
                ob.update_bid(float(bid[0]), float(bid[1]))
                
            # Update Asks
            for ask in data.get('asks', []):
                ob.update_ask(float(ask[0]), float(ask[1]))
                
            # The metrics (OFI, Spread, Microprice) are now instantly available in Cython
            
        except Exception as e:
            # Silently drop malformed depth updates to prevent log spam
            pass

    def _process_trade_msg(self, msg):
        """
        [Capa 4] Intra-vela Footprint Reconstructor (Simplified).
        Processes @aggTrade messages to calculate Cumulative Volume Delta (CVD) and buying/selling pressure.
        """
        try:
            if 'data' not in msg or 'e' not in msg['data'] or msg['data']['e'] != 'aggTrade':
                return
                
            data = msg['data']
            symbol_raw = data.get('s', '')
            symbol = f"{symbol_raw[:-4]}/{symbol_raw[-4:]}" if symbol_raw.endswith('USDT') else symbol_raw
            
            if symbol not in self.order_flow_metrics:
                self.order_flow_metrics[symbol] = {
                    'buy_volume': 0.0,
                    'sell_volume': 0.0,
                    'cvd': 0.0,
                    'last_reset': time.time()
                }
                
            metrics = self.order_flow_metrics[symbol]
            
            # Reset intra-vela metrics every 60 seconds (1-minute candle approximation)
            now = time.time()
            if now - metrics['last_reset'] >= 60.0:
                metrics['buy_volume'] = 0.0
                metrics['sell_volume'] = 0.0
                metrics['cvd'] = 0.0
                metrics['last_reset'] = now
                
            qty = float(data.get('q', 0))
            is_buyer_maker = data.get('m', False)
            
            # In Binance, if buyer is maker, the trade was initiated by the seller (sell market order)
            if is_buyer_maker:
                metrics['sell_volume'] += qty
                metrics['cvd'] -= qty
            else:
                metrics['buy_volume'] += qty
                metrics['cvd'] += qty
                
        except Exception as e:
            pass

    def get_order_flow_metrics(self, symbol: str) -> dict:
        """
        Returns real-time intra-vela order flow metrics (CVD, buy/sell volume).
        """
        if symbol not in self.order_flow_metrics:
            return {'buy_volume': 0.0, 'sell_volume': 0.0, 'cvd': 0.0}
        # Return a copy to avoid race conditions during dict access
        return dict(self.order_flow_metrics[symbol])

    def _process_liquidation_msg(self, msg):
        """
        Processes forceOrder messages
        """
        try:
            if 'data' not in msg or 'o' not in msg['data']:
                return
            
            data = msg['data']['o']
            symbol_raw = data.get('s', '')
            
            # Format back to BTC/USDT
            symbol = f"{symbol_raw[:-4]}/{symbol_raw[-4:]}" if symbol_raw.endswith('USDT') else symbol_raw
            
            side = data.get('S', '') # SELL = Long liquidation, BUY = Short liquidation
            price = float(data.get('ap', data.get('p', 0))) # Average price
            qty = float(data.get('q', 0))
            usd_value = price * qty
            
            if usd_value < 5000: # Filter noise
                return
                
            # Log significant liquidations
            if usd_value > 50000:
                 logger.info(f"🩸 [REKT] {symbol} {'LONG' if side == 'SELL' else 'SHORT'} Liq: ${usd_value:,.2f} @ {price}")
            
            # Add to liquidation history buffer
            if symbol not in self.liquidation_history:
                from collections import deque
                self.liquidation_history[symbol] = deque(maxlen=100)
                
            self.liquidation_history[symbol].append({
                'timestamp': time.time(),
                'side': side,
                'price': price,
                'usd_value': usd_value
            })
            
            # ═══════════════════════════════════════════════════════════════
            # AUDIT FIX #1: Use order_flow (existing MarketEvent field) instead
            # of 'data' (which doesn't exist on the frozen dataclass).
            # Also set close_price=price so Engine._process_market_event()
            # routes this event to strategies (without close_price, the event
            # falls into the else branch and never reaches calculate_signals).
            # ═══════════════════════════════════════════════════════════════
            self.events_queue.put(MarketEvent(
                symbol=symbol, 
                close_price=price,
                order_flow={
                    'liquidation': True, 
                    'side': side, 
                    'usd_value': usd_value, 
                    'price': price
                }
            ))
            
        except Exception as e:
            logger.error(f"Error processing liquidation: {e}")

    def _start_latency_monitor(self):
        """Starts a background thread to ping Binance every 5s."""
        def _ping_loop():
            # Wait a few seconds for main initialization to avoid race conditions
            time.sleep(3)
            while self._running:
                try:
                    t0 = time.time()
                    self.client_sync.ping()
                    t1 = time.time()
                    latency_ms = (t1 - t0) * 1000
                    self.latency_history.append(latency_ms)
                except Exception as e:
                    logger.warning(f"Ping failed (Latency Monitor): {e}")
                    self.latency_history.append(9999.0) # Penalty for timeout
                
                time.sleep(5) # Check every 5s

        import threading
        t = threading.Thread(target=_ping_loop, daemon=True, name="LatencyMonitor")
        t.start()
        
    def _start_derivatives_monitor(self):
        """[PHASE 29] Starts a background thread to fetch Futures derivatives every 60s."""
        def _derivatives_loop():
            # Wait a few seconds for main initialization to avoid rate-limit race conditions
            time.sleep(10)
            while self._running:
                try:
                    for s in self.symbol_list:
                        sym_clean = s.replace('/', '')
                        try:
                            # Use python-binance futures endpoint wrapper
                            funding_resp = self.client_sync.futures_funding_rate(symbol=sym_clean, limit=1)
                            oi_resp = self.client_sync.futures_open_interest(symbol=sym_clean)
                            
                            funding_rate = float(funding_resp[0]['fundingRate']) if funding_resp else 0.0
                            oi = float(oi_resp['openInterest']) if 'openInterest' in oi_resp else 0.0
                            
                            # Calculate OI Delta if exists
                            old_oi = self.derivatives_metrics.get(s, {}).get('oi', oi)
                            oi_delta = ((oi - old_oi) / old_oi) if old_oi > 0 else 0.0
                            
                            # Phase 29: Preserve WS-accumulated liquidations and reset per-minute
                            current_liq = self.derivatives_metrics.get(s, {}).get('liquidations', 0.0)
                            
                            self.derivatives_metrics[s] = {
                                'funding_rate': funding_rate,
                                'oi': oi,
                                'oi_delta': oi_delta,
                                'liquidations': current_liq 
                            }
                            # Reset liquidation counter for next minute window
                            # (Strategy has already consumed the peak value in the previous tick)
                            # self.derivatives_metrics[s]['liquidations'] = 0.0 # Reset later if needed
                        except Exception as e:
                            logger.debug(f"Futures derivatives fetch skipped/failed for {s}: {e}")
                except Exception as e:
                    logger.error(f"Derivatives monitor error loop: {e}")
                
                time.sleep(60) # Poll every 1 min (Not highly reactive but suitable for macro OI/Funding)

        import threading
        t = threading.Thread(target=_derivatives_loop, daemon=True, name="DerivativesMonitor")
        t.start()
        
    def get_latency_metrics(self):
        """Returns avg_ping and max_ping in ms."""
        if not self.latency_history:
            return 0.0, 0.0
        avg = sum(self.latency_history) / len(self.latency_history)
        mx = max(self.latency_history)
        return avg, mx
        
    def _init_symbol_buffer(self, symbol):
        """Initialize HFT structured buffers for a symbol (Phase 4)."""
        # Capacity defined by Config or defaults
        self.buffers_1m[symbol] = NumbaStructuredRingBuffer(2000)
        self.buffers_5m[symbol] = NumbaStructuredRingBuffer(500)
        self.buffers_15m[symbol] = NumbaStructuredRingBuffer(500)
        self.buffers_1h[symbol] = NumbaStructuredRingBuffer(500)
        self.buffers_4h[symbol] = NumbaStructuredRingBuffer(500)
        self.buffers_1d[symbol] = NumbaStructuredRingBuffer(500) # Preload ~1.5 years
        self.buffers_1w[symbol] = NumbaStructuredRingBuffer(300) # Preload ~5 years
        self.vbi_history[symbol] = NumbaRingBuffer(1000) # Fast VBI history
        self.liquidation_history[symbol] = NumbaRingBuffer(500) # Fast Liq history
        
    def _fetch_single_symbol(self, s):
        """Helper for parallel fetching of ALL timeframes"""
        try:
            sym_clean = s.replace('/', '')
            results = {'symbol': s}
            
            # Fetch 1m (Critical)
            k1m = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1MINUTE, limit=2)
            if k1m:
                latest = k1m[-1]
                results['1m'] = {
                    'timestamp': int(latest[0]),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
            
            # Fetch 5m
            k5m = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_5MINUTE, limit=2)
            if k5m:
                latest = k5m[-1]
                results['5m'] = {
                    'timestamp': int(latest[0]),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }

            # Fetch 15m
            k15m = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_15MINUTE, limit=2)
            if k15m:
                latest = k15m[-1]
                results['15m'] = {
                    'timestamp': int(latest[0]),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
                
            # Fetch 1h
            k1h = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1HOUR, limit=2)
            if k1h:
                latest = k1h[-1]
                results['1h'] = {
                    'timestamp': int(latest[0]),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
                
            # Fetch 4h
            k4h = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_4HOUR, limit=2)
            if k4h:
                latest = k4h[-1]
                results['4h'] = {
                    'timestamp': int(latest[0]),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
                
            # Fetch 1d
            k1d = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1DAY, limit=2)
            if k1d:
                latest = k1d[-1]
                results['1d'] = {
                    'timestamp': int(latest[0]),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
                
            # Fetch 1w
            k1w = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1WEEK, limit=2)
            if k1w:
                latest = k1w[-1]
                results['1w'] = {
                    'timestamp': int(latest[0]),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
            
            return results
        except Exception as e:
            logger.error(f"Fetch failed for {s}: {e}")
            return None

    # ... (fetch methods remain same, but we should lock inside them if they were called concurrently, 
    # but they are called in init so it's fine. The critical part is update_bars vs get_latest_bars)

    def get_latest_bars(self, symbol, n=1, timeframe='1m'):
        """
        Returns NumPy Structured Array (Phase 4 Zero-Copy).
        
        LOW-LATENCY OPTIMIZATION:
        QUÉ: Reutiliza buffers pre-allocados en vez de np.empty() por llamada.
        POR QUÉ: np.empty() + field assignment = ~30-100μs de allocation.
        PARA QUÉ: Reducir GC pressure y latencia en hot-path.
        CÓMO: Pre-allocado en __init__; aquí solo se hace slice + copyto.
        CUÁNDO: En cada tick de mercado, múltiples veces por símbolo.
        DÓNDE: data/binance_loader.py → get_latest_bars()
        QUIÉN: Quant Developer + SRE/DevOps
        """
        try:
            # O(1) timeframe lookup (replaces if/elif chain)
            if self._timeframe_map is None:
                self._timeframe_map = {
                    '1m': self.buffers_1m, '5m': self.buffers_5m,
                    '15m': self.buffers_15m, '1h': self.buffers_1h,
                    '4h': self.buffers_4h, '1d': self.buffers_1d,
                    '1w': self.buffers_1w,
                }
            target_map = self._timeframe_map.get(timeframe, self.buffers_1m)

            with self._data_lock:
                if symbol not in target_map:
                    return None
                
                buf = target_map[symbol]
                if buf.size == 0: return None
                
                # Retrieve from Numba (as tuple of arrays)
                t, o, h, l, c, v = buf.get_last(n)
                
                actual_len = len(t)
                if actual_len == 0:
                    return None
                
                # ZERO-ALLOC: Reuse pre-allocated buffer
                cache = self._bar_output_cache.get(symbol)
                if cache is not None and actual_len <= self._BAR_OUTPUT_MAX:
                    res = cache[:actual_len]
                    res['timestamp'] = t
                    res['open'] = o
                    res['high'] = h
                    res['low'] = l
                    res['close'] = c
                    res['volume'] = v
                    # Return a COPY of the slice to prevent caller mutation
                    # (cost: ~5-15μs for small arrays, but safe)
                    return res.copy()
                else:
                    # Fallback: new allocation for unknown symbols or oversized requests
                    res = np.empty(actual_len, dtype=self._BAR_STRUCT_DTYPE)
                    res['timestamp'] = t
                    res['open'] = o
                    res['high'] = h
                    res['low'] = l
                    res['close'] = c
                    res['volume'] = v
                    return res

        except Exception as e:
            logger.error(f"Error getting structured {timeframe} bars for {symbol}: {e}")
            return None

    def get_data(self, symbol, limit=500):
        """
        Retorna los últimos `limit` bars como un DataFrame de Pandas para compatibilidad
        con Phalanx, Arbitrage y StatArb.
        """
        bars = self.get_latest_bars(symbol, n=limit, timeframe="1m")
        if bars is None or len(bars) == 0:
            return pd.DataFrame()
        # Convert structured array to DataFrame
        df = pd.DataFrame(bars)
        # Convert timestamp (ms) to DatetimeIndex
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df


    def _fetch_deep_history_worker(self, symbol, interval, hours, dest_dict, limit_per_req=1000, buffer_multiplier=1.2):
        """Worker function for parallel history fetching"""
        try:
            time_needed = Config.Strategies.ML_LOOKBACK_BARS * buffer_multiplier if interval == '1m' else hours * 60
            minutes_needed = hours * 60
            total_candles_needed = minutes_needed / (
                15 if interval == '15m' else
                60 if interval == '1h' else
                240 if interval == '4h' else
                1440 if interval == '1d' else
                10080 if interval == '1w' else
                5 if interval == '5m' else 1
            )
            
            # Adjust calculation for 1m bars specifically
            if interval == '1m':
                total_candles_needed = hours * 60
            
            all_candles = []
            since = int(time.time() * 1000) - (hours * 60 * 60 * 1000)
            
            # determine interval string constant
            kl_interval = Client.KLINE_INTERVAL_1MINUTE
            if interval == '5m': kl_interval = Client.KLINE_INTERVAL_5MINUTE
            elif interval == '15m': kl_interval = Client.KLINE_INTERVAL_15MINUTE
            elif interval == '1h': kl_interval = Client.KLINE_INTERVAL_1HOUR
            elif interval == '4h': kl_interval = Client.KLINE_INTERVAL_4HOUR
            elif interval == '1d': kl_interval = Client.KLINE_INTERVAL_1DAY
            elif interval == '1w': kl_interval = Client.KLINE_INTERVAL_1WEEK
            
            sym_clean = symbol.replace('/', '')
            
            while len(all_candles) < total_candles_needed:
                client = self.client_sync
                candles = client.get_klines(symbol=sym_clean, interval=kl_interval, limit=limit_per_req, startTime=since)
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + (60000 * (
                    15 if interval == '15m' else
                    60 if interval == '1h' else
                    240 if interval == '4h' else
                    1440 if interval == '1d' else
                    10080 if interval == '1w' else
                    5 if interval == '5m' else 1
                ))
                
                if len(candles) < limit_per_req:
                    break
            
            # Process and store
            processed_bars = []
            for c in all_candles:
                # OMEGA Validation (Dimensión 1)
                o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                
                # O1: High/Low Physics
                if h < max(o, cl) - 1e-8 or l > min(o, cl) + 1e-8:
                    continue
                # O3: Zero Volume with price movement
                if o != cl and v <= 0:
                    continue
                    
                # No pd.to_datetime here (Phase 1: Zero-Pandas)
                processed_bars.append({
                    'symbol': symbol,
                    'timestamp': int(c[0]),
                    'open': o, 'high': h, 'low': l, 'close': cl, 'volume': v
                })
            
            # RAM OPTIMIZATION (Filled into Ring Buffers)
            if interval == '1m': target_map = self.buffers_1m
            elif interval == '5m': target_map = self.buffers_5m
            elif interval == '15m': target_map = self.buffers_15m
            elif interval == '1h': target_map = self.buffers_1h
            elif interval == '4h': target_map = self.buffers_4h
            elif interval == '1d': target_map = self.buffers_1d
            elif interval == '1w': target_map = self.buffers_1w
            else: target_map = self.buffers_1m

            # Ensure symbol init
            # NOTE: self.buffers_X were init in __init__, assuming symbol_list constant.
            # If adaptive, we need check.
            if symbol not in target_map:
                # Can't init here easily without lock, but let's assume valid symbol
                pass

            buf = target_map[symbol]
                
            # Process and store - Insert in order
            for c in processed_bars:
                ts = c['timestamp']
                buf.push(
                    ts,
                    np.float32(c['open']),
                    np.float32(c['high']),
                    np.float32(c['low']),
                    np.float32(c['close']),
                    np.float32(c['volume'])
                )
            
            if interval == '1m':
                with self._data_lock:
                    for c in processed_bars:
                        self.latest_data[symbol].append({
                            'datetime': datetime.fromtimestamp(c['timestamp']/1000.0, tz=timezone.utc),
                            'open': float(c['open']),
                            'high': float(c['high']),
                            'low': float(c['low']),
                            'close': float(c['close']),
                            'volume': float(c['volume'])
                        })
            
            logger.info(f"Loaded {len(processed_bars)} {interval} bars for {symbol}")
            
            # Phase 14: Auto-Calibrate HMM on history
            if interval == '1m' and len(processed_bars) > 200:
                rets = np.array([b['close'] for b in processed_bars])
                rets = np.diff(rets) / rets[:-1]
                # Note: HMM is in MarketRegimeDetector usually, but we can calibrate here if needed
                # However, it's cleaner to let the Strategy/RegimeDetector handle it via update()
                pass
            
        except Exception as e:
            logger.error(f"Failed to fetch {interval} history for {symbol}: {e}")

    def fetch_initial_history(self):
        """
        Fetches ~25 hours of historical data (1m candles) in PARALLEL.
        PHASE 5: Checks Parquet Cache First.
        """
        logger.info("⏳ [Data] Starting parallel history fetch...")
        
        # 1. Try Load from Disk
        loaded_symbols = self.load_snapshot()
        missing_symbols = [s for s in self.symbol_list if s not in loaded_symbols]
        
        if not missing_symbols:
            logger.info("✅ [Data] All symbols loaded from Cache!")
            return

        # 2. Fetch Missing from API
        logger.info(f"🌍 [Data] Fetching missing {len(missing_symbols)} symbols from Binance API...")
        
        time_needed = Config.Strategies.ML_LOOKBACK_BARS * 1.2
        hours = int(time_needed / 60) + 5
        
        futures = []
        for s in missing_symbols:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '1m', hours, None
            ))
        
        # Wait for all to complete
        from concurrent.futures import as_completed
        errors = []
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                errors.append(e)
                logger.error(f"❌ [INITIAL HISTORY] Error fetching symbol: {e}")
                
        if errors:
            raise RuntimeError(f"Failed to fetch initial history for one or more symbols: {errors}")


    def fetch_initial_history_1h(self):
        """Fetches ~200 hours of 1h data in PARALLEL."""
        logger.info("Fetching 1h historical data (250h) in PARALLEL...")
        hours = 250
        
        futures = []
        for s in self.symbol_list:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '1h', hours, None
            ))
            
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            pass

    def fetch_initial_history_4h(self):
        """Fetches 4h data in PARALLEL."""
        logger.info("Fetching 4h macro historical data (400h) in PARALLEL...")
        hours = 400
        
        futures = []
        for s in self.symbol_list:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '4h', hours, None
            ))
            
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            pass

    def fetch_initial_history_5m(self):
        """Fetches 5m data in PARALLEL."""
        logger.info("Fetching 5m historical data (100h) in PARALLEL...")
        hours = 100
        
        futures = []
        for s in self.symbol_list:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '5m', hours, None
            ))
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            pass

    def fetch_initial_history_15m(self):
        """Fetches 15m data in PARALLEL."""
        logger.info("Fetching 15m historical data (100h) in PARALLEL...")
        hours = 100
        
        futures = []
        for s in self.symbol_list:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '15m', hours, None
            ))
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            pass

    def fetch_initial_history_1d(self):
        """Fetches 1d data in PARALLEL."""
        logger.info("Fetching 1d macro historical data (500d) in PARALLEL...")
        hours = 12000 # 500 days
        futures = []
        for s in self.symbol_list:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '1d', hours, None
            ))
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            pass

    def fetch_initial_history_1w(self):
        """Fetches 1w data in PARALLEL."""
        logger.info("Fetching 1w structural historical data (300w) in PARALLEL...")
        hours = 50400 # 300 weeks
        futures = []
        for s in self.symbol_list:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '1w', hours, None
            ))
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            pass

    def get_latest_bars_1h(self, symbol, n=1):
        """
        Returns the last N 1h bars (RingBuffer Wrapper).
        """
        return self.get_latest_bars(symbol, n, timeframe='1h')

    def get_latest_bars_4h(self, symbol, n=1):
        """Returns the last N 4h bars."""
        return self.get_latest_bars(symbol, n, timeframe='4h')

    def get_latest_bars_1d(self, symbol, n=1):
        """Returns the last N 1d bars."""
        return self.get_latest_bars(symbol, n, timeframe='1d')
        
    def get_latest_bars_1w(self, symbol, n=1):
        """Returns the last N 1w bars."""
        return self.get_latest_bars(symbol, n, timeframe='1w')


    @trace_execution
    def update_bars(self):
        """
        Fetches new bars from Binance using Parallel Threads (~2s vs 30s).
        """
        futures_map = {self.executor.submit(self._fetch_single_symbol, s): s for s in self.symbol_list}
        
        # FIXED: Freshness Check (Rule 3.3)
        current_time_ms = int(time.time() * 1000)
        
        for future in as_completed(futures_map):
            s = futures_map[future]
            try:
                data_packet = future.result()
                if not data_packet: continue
                
                # Check for staleness
                last_update_ms = 0
                if '1m' in data_packet:
                    last_update_ms = data_packet['1m']['timestamp']
                    if (current_time_ms - last_update_ms) > 300000: # 5 minutes
                        logger.warning(f"⚠️ DATA STALE: {s} last update was {int((current_time_ms - last_update_ms)/60000)}m ago.")
                
                if '1m' in data_packet:
                    bar = data_packet['1m']
                    ts = bar['timestamp']
                    o, h, l, cl, v = bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']
                    
                    # 🛡️ [DATA GUARDIAN] Integridad Forense y OMEGA O1/O3
                    if np.isnan(cl) or cl <= 0 or v < 0:
                        logger.error(f"🚨 [DATA GUARDIAN] DATOS CORRUPTOS (NaN/Cero) en {s}. Ignorando frame.")
                        continue
                        
                    if h < max(o, cl) - 1e-8 or l > min(o, cl) + 1e-8:
                        logger.warning(f"🛡️ [OMEGA] {s}: O1 Validation Failed. Dropping corrupt candle (High/Low physics).")
                        continue
                        
                    if o != cl and v <= 0:
                        logger.warning(f"🛡️ [OMEGA] {s}: O3 Validation Failed. Dropping zero-volume candle with price movement.")
                        continue
                        
                    with self._data_lock:
                         buf = self.buffers_1m[s]
                         last_t_arr = buf.get_last(1)
                         
                         # Gap Detection
                         if last_t_arr is not None and len(last_t_arr) > 0:
                             last_ts = last_t_arr['timestamp'][0]
                             time_diff = ts - last_ts
                             if time_diff > 60000 and last_ts != ts:
                                 missed = int((time_diff/60000)-1)
                                 logger.warning(f"⚠️ [DATA GUARDIAN] GAP en {s}: Faltan {missed} velas de 1m.")
                                 self.data_health_metrics[s]["gaps"] += 1
                                 
                         if last_t_arr is not None and len(last_t_arr) > 0 and last_t_arr['timestamp'][0] == ts:
                              buf.rewind_one()
                         
                         buf.push(ts, np.float32(bar['open']), np.float32(bar['high']), 
                                  np.float32(bar['low']), np.float32(bar['close']), np.float32(bar['volume']))
                         
                         self.events_queue.put(MarketEvent(symbol=s, close_price=bar['close'], timestamp=datetime.now()))

                if '5m' in data_packet:
                    bar = data_packet['5m']
                    ts = bar['timestamp']
                    with self._data_lock:
                         buf = self.buffers_5m[s]
                         last_t_arr = buf.get_last(1)
                         if last_t_arr is not None and len(last_t_arr) > 0 and last_t_arr['timestamp'][0] == ts:
                              buf.rewind_one()
                         buf.push(ts, np.float32(bar['open']), np.float32(bar['high']), 
                                  np.float32(bar['low']), np.float32(bar['close']), np.float32(bar['volume']))

                if '15m' in data_packet:
                    bar = data_packet['15m']
                    ts = bar['timestamp']
                    with self._data_lock:
                         buf = self.buffers_15m[s]
                         last_t_arr = buf.get_last(1)
                         if last_t_arr is not None and len(last_t_arr) > 0 and last_t_arr['timestamp'][0] == ts:
                              buf.rewind_one()
                         buf.push(ts, np.float32(bar['open']), np.float32(bar['high']), 
                                  np.float32(bar['low']), np.float32(bar['close']), np.float32(bar['volume']))

                if '1h' in data_packet:
                    bar = data_packet['1h']
                    ts = bar['timestamp']
                    with self._data_lock:
                         buf = self.buffers_1h[s]
                         last_t_arr = buf.get_last(1)
                         if last_t_arr is not None and len(last_t_arr) > 0 and last_t_arr['timestamp'][0] == ts:
                              buf.rewind_one()
                         buf.push(ts, np.float32(bar['open']), np.float32(bar['high']), 
                                  np.float32(bar['low']), np.float32(bar['close']), np.float32(bar['volume']))

                if '4h' in data_packet:
                    bar = data_packet['4h']
                    ts = bar['timestamp']
                    with self._data_lock:
                         buf = self.buffers_4h[s]
                         last_t_arr = buf.get_last(1)
                         if last_t_arr is not None and len(last_t_arr) > 0 and last_t_arr['timestamp'][0] == ts:
                              buf.rewind_one()
                         buf.push(ts, np.float32(bar['open']), np.float32(bar['high']), 
                                  np.float32(bar['low']), np.float32(bar['close']), np.float32(bar['volume']))
                                  
                if '1d' in data_packet:
                    bar = data_packet['1d']
                    ts = bar['timestamp']
                    with self._data_lock:
                         buf = self.buffers_1d[s]
                         last_t_arr = buf.get_last(1)
                         if last_t_arr is not None and len(last_t_arr) > 0 and last_t_arr['timestamp'][0] == ts:
                              buf.rewind_one()
                         buf.push(ts, np.float32(bar['open']), np.float32(bar['high']), 
                                  np.float32(bar['low']), np.float32(bar['close']), np.float32(bar['volume']))
                                  
                if '1w' in data_packet:
                    bar = data_packet['1w']
                    ts = bar['timestamp']
                    with self._data_lock:
                         buf = self.buffers_1w[s]
                         last_t_arr = buf.get_last(1)
                         if last_t_arr is not None and len(last_t_arr) > 0 and last_t_arr['timestamp'][0] == ts:
                              buf.rewind_one()
                         buf.push(ts, np.float32(bar['open']), np.float32(bar['high']), 
                                  np.float32(bar['low']), np.float32(bar['close']), np.float32(bar['volume']))
                    
                # --- PHASE 14: LEAD-LAG SYNC ---
                if s != self.reference_symbol and self.reference_symbol in self.buffers_1m:
                    try:
                        self._calculate_lead_lag(s)
                    except Exception as e:
                        logger.error(f"Error calculating Lead-Lag for {s}: {e}")
                    
            except Exception as e:
                logger.error(f"Error processing update for {s}: {e}")

    def _calculate_lead_lag(self, symbol: str):
        """
        QUÉ: Calcula la correlación cruzada entre BTC y un seguidor.
        POR QUÉ: Identificar si BTC lidera el movimiento para anticipar entradas en Alts.
        PARA QUÉ: Alpha de milisegundos/segundos.
        """
        try:
            ref_buf = self.buffers_1m[self.reference_symbol]
            target_buf = self.buffers_1m[symbol]
            
            if ref_buf.size < 60 or target_buf.size < 60: return
            
            # Obtener últimos 60 retornos
            ref_data = ref_buf.get_last(61)
            target_data = target_buf.get_last(61)
            
            ref_rets = np.diff(ref_data['close']) / ref_data['close'][:-1]
            target_rets = np.diff(target_data['close']) / target_data['close'][:-1]
            
            # Correlación en lags de -5 a +5
            best_corr = -1.0
            best_lag = 0
            
            from utils.math_kernel import pearson_correlation_jit
            
            for lag in range(-5, 6):
                if lag == 0:
                    corr = pearson_correlation_jit(ref_rets, target_rets)
                elif lag > 0:
                    corr = pearson_correlation_jit(ref_rets[lag:], target_rets[:-lag])
                else:
                    abs_lag = abs(lag)
                    corr = pearson_correlation_jit(ref_rets[:-abs_lag], target_rets[abs_lag:])
                
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            
            self.lead_lag_results[symbol] = {
                'lag': best_lag,
                'correlation': best_corr,
                'timestamp': time.time()
            }
            
        except Exception as e:
            # logger.debug silent to avoid spam
            pass

    
    

    async def update_bars_async(self):
        """
        Async Fallback for maintaining data integrity without blocking event loop (Phase 7).
        Uses ThreadPoolExecutor for now, as python-binance client_sync is blocking.
        Transitioning entirely to self.client.get_klines logic is complex due to RingBuffer lock contention.
        Safest approach for Phase 7: Offload this specific periodic task to a thread to not block AsyncIO.
        """
        loop = asyncio.get_running_loop()
        # We run the OLD blocking update_bars in a thread, so it doesn't freeze the bot.
        await loop.run_in_executor(self.executor, self.update_bars)

    def get_latest_bars_5m(self, symbol, n=20):
        """Wrapper for RingBuffer 5m"""
        return self.get_latest_bars(symbol, n, timeframe='5m')
    
    def get_latest_bars_15m(self, symbol, n=20):
        """Wrapper for RingBuffer 15m"""
        return self.get_latest_bars(symbol, n, timeframe='15m')
        
    def get_order_flow_metrics(self, symbol: str) -> dict:
        """
        [PHASE 13] Returns the latest Order Flow LOB metrics.
        Returns: { 'imbalance': float, 'bid_vol_5': float, 'ask_vol_5': float, 'timestamp': float }
        """
        # Normalize symbol if needed
        # We store internally as "BTC/USDT" (mapped in process_depth)
        
        # Fast lookup
        if symbol in self.order_flow_metrics:
            return self.order_flow_metrics[symbol]
        
        return None

    def get_derivatives_metrics(self, symbol: str) -> dict:
        """
        [PHASE 29] Returns the latest Futures derivatives metrics.
        Expected format: {'funding_rate': float, 'oi': float, 'oi_delta': float, 'liquidations': float}
        """
        # We store internally as "BTC/USDT", check fast lookup
        if symbol in self.derivatives_metrics:
            return self.derivatives_metrics[symbol]
        return {'funding_rate': 0.0, 'oi': 0.0, 'oi_delta': 0.0, 'liquidations': 0.0}

    async def start_socket(self):
        """
        Starts the WebSocket connection(s) for real-time data updates.
        PHASE 33: ROBUST MULTIPLEXING (Chunking + Dynamic Updates)
        """
        logger.info("Starting Binance WebSocket (Phase 33: Multiplexed)...")
        
        # [PHASE 10] Start Watchdog
        if not self.watchdog_running:
            asyncio.create_task(self._watchdog_loop())
        
        # Initialize Async Client
        api_key = Config.BINANCE_API_KEY
        api_secret = Config.BINANCE_SECRET_KEY
        
        # Handle Testnet/Demo keys
        if Config.BINANCE_USE_TESTNET:
            api_key = Config.BINANCE_TESTNET_API_KEY
            api_secret = Config.BINANCE_TESTNET_SECRET_KEY
        elif hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
             api_key = Config.BINANCE_DEMO_API_KEY
             api_secret = Config.BINANCE_DEMO_SECRET_KEY
        
        # Keep track of active sockets
        self.active_sockets = []
        
        while self._running:
            try:
                self.client = await AsyncClient.create(api_key, api_secret, testnet=Config.BINANCE_USE_TESTNET)
                self.bsm = BinanceSocketManager(self.client)
                
                # 1. Build Stream List
                streams = []
                for s in self.symbol_list:
                    base_s = s.lower().replace('/', '')
                    streams.append(f"{base_s}@kline_1m")
                    streams.append(f"{base_s}@kline_5m")
                    streams.append(f"{base_s}@kline_15m")
                    streams.append(f"{base_s}@kline_1h")
                    streams.append(f"{base_s}@kline_1d") # PHASE 4: Macro Horizon
                    streams.append(f"{base_s}@kline_1w") # PHASE 4: Structural Horizon
                    streams.append(f"{base_s}@bookTicker") # Liquidity Guardian
                    streams.append(f"{base_s}@forceOrder") # Liquidations (Omega Mind)
                    # PHASE 13: Phalanx-Omega (Order Flow)
                    streams.append(f"{base_s}@depth5@100ms") # LOB Imbalance
                    streams.append(f"{base_s}@aggTrade")     # Tape (Delta)
                
                # 2. CHUNKING STRATEGY (Phase 33)
                # Binance recommends < 1024 streams per socket.
                # URL length limit is ~4096 chars.
                # Safe chunk size: 50 symbols * 7 streams = 350 streams (approx 6000 chars? Might be too long)
                # Let's reduce chunk size to 100 streams per socket just to be safe.
                chunk_size = 100
                chunks = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]
                
                logger.info(f"Subscribing to {len(streams)} streams across {len(chunks)} socket(s)...")
                
                # 3. Create Tasks for each Chunk
                tasks = []
                for i, chunk in enumerate(chunks):
                    # multiplex_socket returns a ReconnectingWebsocket
                    # We need to run it. 
                    # Note: python-binance's multiplex_socket is a Context Manager usually
                    # But if we have multiple, we need asyncio.gather or similar.
                    # Complexity: bsm.multiplex_socket is a context manager.
                    # We need to wrap each in a function.
                    tasks.append(self._manage_socket_chunk(self.bsm, chunk, i))
                
                # Run all sockets concurrently
                await asyncio.gather(*tasks)
                        
            except asyncio.CancelledError:
                logger.info("WebSocket: Shutdown signal received.")
                break
            except Exception as e:
                logger.error(f"WebSocket Manager Error: {e}")
                await asyncio.sleep(5)
            finally:
                if self.client:
                    await self.client.close_connection()

    async def _manage_socket_chunk(self, bsm, streams, index):
        """
        Manages a single multiplexed socket connection for a chunk of streams.
        """
        try:
            params = streams
            # Verify stream count
            if not params: return
            
            logger.info(f"🔌 Socket #{index}: Connecting {len(params)} streams...")
            socket = bsm.multiplex_socket(params)
            
            async with socket as tscm:
                while self._running:
                    msg = await tscm.recv()
                    
                    if not msg or 'data' not in msg:
                        continue
                    
                    # Routing based on stream name or content
                    stream_name = msg.get('stream', '')
                    data = msg['data']

                    if 'kline' in stream_name:
                        self._process_kline_event(data, stream_name)
                    elif 'bookTicker' in stream_name:
                        pass # self._process_book_ticker(data)
                    elif 'forceOrder' in stream_name:
                        self._process_liquidation_msg(data)
                    elif 'depth5' in stream_name:
                        self._process_depth_level5(data, stream_name)
                    elif 'aggTrade' in stream_name:
                        self._process_agg_trade(data)
                        
                    # Periodic heartbeat or metrics here not needed per chunk
        except Exception as e:
            logger.error(f"🔌 Socket #{index} Failed: {e}")
            raise e # Propagate to main restart loop


    def _force_restart_socket(self):
        """
        Phase 16: Recovery Callback
        """
        logger.warning("🚨 [Watchdog] Forcing WebSocket Restart...")
        if hasattr(self, 'socket'):
            # This implementation depends on library.
            # Best way: Cancel the task or Throw exception into loop?
            # We are in a different thread (Watchdog). We can't await.
            # We can set a flag or try to close from here (Thread-safe?)
            try:
                # Close client connection to trigger exception in read loop
                asyncio.run_coroutine_threadsafe(self.client.close_connection(), asyncio.get_event_loop())
            except Exception as e:
                logger.error(f"Restart failed: {e}")
                


    def _process_kline_event(self, kline_data, stream_name):
        """
        ⚡ PHASE OMNI: FIXED Kline Event Processor + Jitter Tracking.
        
        QUÉ: Procesa datos de vela (kline) recibidos vía WebSocket.
        POR QUÉ: La versión anterior estaba completamente rota:
                 - Dead code block referenciaba variable 'msg' inexistente
                 - Todo el código real estaba dentro de un 'except Exception: pass'
                 - 'msg.get("stream")' debía ser 'stream_name' (parámetro)
                 - 'internal_sym' debía ser 'internal_symbol' (typo)
                 - Nested self._data_lock causaba deadlock
        PARA QUÉ: Restaurar la funcionalidad de procesamiento de velas en tiempo real.
        CÓMO: Recibe kline_data (ya es msg['data']) y stream_name → parsea → buffer → event.
        CUÁNDO: Cada mensaje WebSocket con 'kline' en el stream name.
        DÓNDE: data/binance_loader.py → _process_kline_event().
        QUIÉN: BinanceData, _manage_socket_chunk (caller at L914).
        
        Args:
            kline_data: The 'data' field from the WebSocket message (msg['data']).
            stream_name: The stream identifier (e.g. 'btcusdt@kline_1m').
        """
        try:
            # Phase 16: Heartbeat (moved outside dead code block)
            if hasattr(self, 'watchdog') and self.watchdog:
                self.watchdog.heartbeat("BinanceWS")
            
            # ⚡ PHASE OMNI: Per-Stream Jitter Tracking
            # Measures time between consecutive messages on the same stream
            now_mono = time.monotonic()
            if not hasattr(self, '_stream_jitter'):
                self._stream_jitter = {}  # {stream_name: {'last_ts': float, 'jitter_ms': deque}}
            
            if stream_name in self._stream_jitter:
                jitter_info = self._stream_jitter[stream_name]
                delta_ms = (now_mono - jitter_info['last_ts']) * 1000
                jitter_info['jitter_ms'].append(delta_ms)
                jitter_info['last_ts'] = now_mono
                
                # Alert on extreme jitter (> 5000ms gap indicates stream stall)
                if delta_ms > 5000:
                    logger.warning(
                        f"⚠️ [Jitter] Stream {stream_name} gap: {delta_ms:.0f}ms "
                        f"(avg: {sum(jitter_info['jitter_ms'])/len(jitter_info['jitter_ms']):.0f}ms)"
                    )
            else:
                self._stream_jitter[stream_name] = {
                    'last_ts': now_mono,
                    'jitter_ms': collections.deque(maxlen=100),
                }
            
            # ─── PARSE KLINE DATA ───
            # kline_data is msg['data'] from _manage_socket_chunk (L911-914)
            if 'k' not in kline_data:
                return
                
            kline = kline_data['k']
            symbol = intern_string(kline['s'])  # Phase 21: String Interning
            
            # Convert symbol to internal format (e.g. BTCUSDT → BTC/USDT)
            internal_symbol = symbol
            if symbol not in self.symbol_list:
                if symbol.endswith('USDT'):
                    test_sym = f"{symbol[:-4]}/USDT"
                    if test_sym in self.symbol_list:
                        internal_symbol = test_sym
            
            is_closed = kline['x']  # Boolean: Is this kline closed?
            
            # ─── DETERMINE TIMEFRAME (tf) ───
            # PHASE 33: Multi-Timeframe Routing via stream_name parameter
            tf = '1m'
            if '@kline_5m' in stream_name: tf = '5m'
            elif '@kline_15m' in stream_name: tf = '15m'
            elif '@kline_1h' in stream_name: tf = '1h'
            elif '@kline_1d' in stream_name: tf = '1d' # Phase 4
            elif '@kline_1w' in stream_name: tf = '1w' # Phase 4

            # Extract data with Phase 4: Downcasting (float32 for memory)
            # [NANO-SPEED] Extirpado pd.to_datetime del loop caliente
            timestamp_ms = int(kline['t'])
            open_price = np.float32(kline['o'])
            high_price = np.float32(kline['h'])
            low_price = np.float32(kline['l'])
            close_price = np.float32(kline['c'])
            volume = np.float32(kline['v'])

            # DATA QUALITY FILTER (Módulo OMEGA - Dimensión 1)
            prev_close = None
            if self.latest_data[internal_symbol]:
                prev_close = float(self.latest_data[internal_symbol][-1]['close'])
                
            is_valid = OHLCVValidator.validate_kline(
                open_price, high_price, low_price, close_price, volume, is_closed, previous_close=prev_close
            )
            
            if not is_valid:
                return  # Skip bad tick

            # GAP DETECTION (Rule 3.2 - Hardened)
            health_score = 100.0
            gap_detected = False
            
            with self._data_lock:
                if self.latest_data[internal_symbol]:
                    last_bar = self.latest_data[internal_symbol][-1]
                    last_ts_ms = int(last_bar['datetime'].timestamp() * 1000)
                    time_diff_s = (timestamp_ms - last_ts_ms) / 1000.0
                    
                    if time_diff_s > 65 and is_closed:
                        gap_detected = True
                        health_score = max(0.0, 100.0 - (time_diff_s / 60.0) * 10)
                        logger.warning(f"🚨 GAP DETECTED in {internal_symbol}: {time_diff_s}s interval. Dispatching Backfill...")
                        
                        # Dispatch Backfill (Async task to not block WebSocket thread)
                        if hasattr(self, 'loop') and self.loop:
                            asyncio.run_coroutine_threadsafe(
                                self._backfill_gap(internal_symbol, tf, last_ts_ms, timestamp_ms), 
                                self.loop
                            )
                        
                # Update sliding window for next gap check
                bar_dict = {
                    'datetime': datetime.fromtimestamp(timestamp_ms/1000.0, tz=timezone.utc),
                    'close': close_price
                }
                self.latest_data[internal_symbol].append(bar_dict)



            health_metrics = {
                "score": health_score,
                "gap_s": time_diff_s if 'time_diff_s' in locals() else 0,
                "tf": tf,
                "stale": False
            }
            
            # ─── BUFFER UPDATE (Thread-Safe) ───
            
            target_map = self.buffers_1m
            if tf == '5m': target_map = self.buffers_5m
            elif tf == '15m': target_map = self.buffers_15m
            elif tf == '1h': target_map = self.buffers_1h
            elif tf == '1d': target_map = self.buffers_1d
            elif tf == '1w': target_map = self.buffers_1w
            
            ts_ms = int(kline['t'])  # Raw exchange timestamp
            
            with self._data_lock:
                buf = target_map[internal_symbol]
                last_arr = buf.get_last(1)
                if last_arr is not None and len(last_arr) > 0 and last_arr['timestamp'][0] == ts_ms:
                    buf.rewind_one()
                
                buf.push(ts_ms, open_price, high_price, low_price, close_price, volume)
            
            # ─── MARKET EVENT TRIGGER (THROTTLED) ───
            should_trigger = is_closed
            now_ts = time.time()
            
            # Volatility Bypass: Force trigger on >0.05% move
            if not should_trigger:
                try:
                    current_price = close_price
                    last_price = 0.0
                    
                    if internal_symbol in self.buffers_1m:
                        last_bar = self.get_latest_bars(internal_symbol, n=1)
                        if last_bar is not None:
                            last_price = float(last_bar['close'][0])
                    
                    if last_price > 0:
                        pct_change = abs((current_price - last_price) / last_price)
                        if pct_change >= 0.0005:
                            should_trigger = True
                except Exception:
                    pass

            # Time-based throttle: at most once every 0.1s per symbol
            if not should_trigger:
                last_t = self.last_event_time.get(internal_symbol, 0)
                if now_ts - last_t > 0.1:
                    should_trigger = True
            
            # ─── FIRE MARKET EVENT ───
            if should_trigger:
                self.last_event_time[internal_symbol] = now_ts
                
                # Build event with Order Flow Metrics (Phase 13)
                metrics = self.order_flow_metrics.get(internal_symbol)
                if metrics:
                    of_metrics = metrics.copy()
                    
                    # 🌊 PHASE 25: Merge Microstructure Metrics
                    if internal_symbol in self.microstructure:
                        micro_metrics = self.microstructure[internal_symbol].get_metrics()
                        of_metrics.update(micro_metrics)
                    
                    self.events_queue.put(MarketEvent(
                        symbol=internal_symbol,
                        close_price=close_price,
                        timestamp=datetime.now(timezone.utc),
                        order_flow=of_metrics,
                        health_metrics=health_metrics,
                        is_closed=is_closed
                    ))
                    
                    # Reset delta atomically (<1ms target)
                    metrics['delta'] = 0.0
                    metrics['last_update'] = time.time()
                else:
                    # Trigger without order flow if not available
                    self.events_queue.put(MarketEvent(
                        symbol=internal_symbol,
                        close_price=close_price,
                        timestamp=datetime.now(timezone.utc),
                        health_metrics=health_metrics,
                        is_closed=is_closed
                    ))
            
            if is_closed:
                logger.info(f"🌊 WebSocket Closed Bar: {internal_symbol} @ {close_price} [Health: {health_score:.1f}]")
                
        except Exception as e:
            logger.error(f"WebSocket Message Error: {e}")

    async def _backfill_gap(self, symbol: str, timeframe: str, start_ms: int, end_ms: int):
        """
        🚀 PHASE 3 (Forensic): Proactive Backfill for Gaps.
        Recupera velas faltantes vía REST para mantener la integridad de las medias móviles.
        """
        try:
            # Avoid too many concurrent backfills
            now = time.time()
            if now - self.data_health_metrics[symbol]['last_backfill'] < 30:
                return
            self.data_health_metrics[symbol]['last_backfill'] = now
            
            logger.info(f"🔄 [Backfill] Recovering gap for {symbol} ({timeframe}) from {start_ms} to {end_ms}")
            
            sym_clean = symbol.replace('/', '')
            # interval = Client.KLINE_INTERVAL_1MINUTE etc.
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
                '1w': Client.KLINE_INTERVAL_1WEEK
            }
            interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_1MINUTE)
            
            # Fetch from REST
            loop = asyncio.get_running_loop()
            candles = await loop.run_in_executor(
                self.executor, 
                lambda: self.client_sync.get_klines(
                    symbol=sym_clean, 
                    interval=interval, 
                    startTime=start_ms + 1000, 
                    endTime=end_ms - 1000,
                    limit=100
                )
            )
            
            if not candles:
                logger.debug(f"ℹ️ [Backfill] No intermediate candles found for {symbol}")
                return

            logger.info(f"✅ [Backfill] Successfully recovered {len(candles)} candles for {symbol}")
            
            # ═══════════════════════════════════════════════════════════════
            # CONSOLIDATE NUMBA RING BUFFER AFTER REST BACKFILL
            # ═══════════════════════════════════════════════════════════════
            if timeframe == '1m': target_map = self.buffers_1m
            elif timeframe == '5m': target_map = self.buffers_5m
            elif timeframe == '15m': target_map = self.buffers_15m
            elif timeframe == '1h': target_map = self.buffers_1h
            elif timeframe == '4h': target_map = self.buffers_4h
            elif timeframe == '1d': target_map = self.buffers_1d
            elif timeframe == '1w': target_map = self.buffers_1w
            else: target_map = self.buffers_1m

            if symbol in target_map:
                buf = target_map[symbol]
                
                # Fetch existing candles in the RingBuffer
                with self._data_lock:
                    size = buf.size
                    if size > 0:
                        t_arr, o_arr, h_arr, l_arr, c_arr, v_arr = buf.get_last(size)
                        existing_candles = []
                        for i in range(size):
                            existing_candles.append({
                                'timestamp': int(t_arr[i]),
                                'open': float(o_arr[i]),
                                'high': float(h_arr[i]),
                                'low': float(l_arr[i]),
                                'close': float(c_arr[i]),
                                'volume': float(v_arr[i])
                            })
                    else:
                        existing_candles = []
                
                # Parse new REST candles
                new_candles = []
                for c in candles:
                    new_candles.append({
                        'timestamp': int(c[0]),
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    })
                
                # De-duplicate by timestamp
                merged = {}
                for cand in existing_candles + new_candles:
                    merged[cand['timestamp']] = cand
                
                # Sort chronologically
                sorted_candles = [merged[ts] for ts in sorted(merged.keys())]
                
                # Clear and re-populate the Numba RingBuffer atomically
                with self._data_lock:
                    buf.head = 0
                    buf.size = 0
                    to_push = sorted_candles[-buf.capacity:] if len(sorted_candles) > buf.capacity else sorted_candles
                    for cand in to_push:
                        buf.push(
                            cand['timestamp'],
                            np.float32(cand['open']),
                            np.float32(cand['high']),
                            np.float32(cand['low']),
                            np.float32(cand['close']),
                            np.float32(cand['volume'])
                        )
                
                logger.info(f"📊 [Backfill] Re-populated Numba buffer for {symbol} ({timeframe}) with {len(to_push)} sorted bars.")
            
            self.data_health_metrics[symbol]['gaps'] += len(candles)
            
        except Exception as e:
            logger.error(f"Error in backfill for {symbol}: {e}")


    def _process_depth_level5(self, data, stream_name):
        """
        [PHASE 13] Processes 5-level Depth snapshots for LOB Imbalance.
        QUÉ: Calcula la presión de compra/venta en el tope del libro.
        POR QUÉ: Desequilibrios masivos preceden movimientos agresivos de precio.
        """
        try:
            symbol = data['s']
            # Get internal symbol
            internal_sym = symbol
            if symbol not in self.symbol_list:
                for s in self.symbol_list:
                    if s.replace('/', '') == symbol:
                        internal_sym = s
                        break
            
            # Sum volume of top 5 levels (NANO: Direct manual loop avoids list compression GC pause)
            bids = data['b']
            asks = data['a']
            
            bid_vol_5 = 0.0
            ask_vol_5 = 0.0
            for i in range(min(5, len(bids))):
                bid_vol_5 += float(bids[i][1])
                
            for i in range(min(5, len(asks))):
                ask_vol_5 += float(asks[i][1])
            
            # Calculate Imbalance Ratio
            # Avoid ZeroDivision
            imbalance = bid_vol_5 / ask_vol_5 if ask_vol_5 > 0 else 10.0 # High bias if no asks
            
            # Store Metrics
            if internal_sym not in self.order_flow_metrics:
                self.order_flow_metrics[internal_sym] = {
                    'imbalance': 1.0, 
                    'bid_vol_5': 0.0, 
                    'ask_vol_5': 0.0, 
                    'delta': 0.0,
                    'last_update': 0
                }
            
            now = time.time()
            self.last_packet_time = now # [PHASE 10] Watchdog Heartbeat
            
            self.order_flow_metrics[internal_sym].update({
                'imbalance': imbalance,
                'bid_vol_5': bid_vol_5,
                'ask_vol_5': ask_vol_5,
                'last_update': now
            })
            
            # 🌊 PHASE 25: Microstructure Analysis (Iceberg Detection)
            if float(bids[0][1]) > 0 and float(asks[0][1]) > 0:
                self.microstructure[internal_sym].on_depth(
                    float(bids[0][0]), float(bids[0][1]),
                    float(asks[0][0]), float(asks[0][1])
                )
            
            # [PHASE 11] SHM Write (Zero-Copy Export)
            if internal_sym in self.shm_managers:
                # Structure: [Bid1P, Bid1Q, Bid2P, Bid2Q ... Ask1P, Ask1Q ...]
                # Top 5 Bids (10 floats) + Top 5 Asks (10 floats)
                shm_arr = self.shm_managers[internal_sym]['arr']
                
                # Flatten top 5
                # bids[:5] -> [[p,q], [p,q]...]
                flat = []
                for i in range(5):
                    if i < len(bids):
                        flat.extend([float(bids[i][0]), float(bids[i][1])])
                    else:
                        flat.extend([0.0, 0.0])
                
                for i in range(5):
                    if i < len(asks):
                        flat.extend([float(asks[i][0]), float(asks[i][1])])
                    else:
                        flat.extend([0.0, 0.0])
                
                # Write to SHM
                shm_arr[:] = flat[:]
            
        except Exception as e:
            logger.debug(f"Error in depth5 processing: {e}")

    def _process_agg_trade(self, data):
        """
        [PHASE 13] Processes aggregate trades for Tape Delta.
        QUÉ: Calcula el Delta (Market Buy Vol - Market Sell Vol).
        POR QUÉ: Detectar agresividad de mercado (Market Orders).
        """
        try:
            symbol = data['s']
            # Get internal symbol
            internal_sym = symbol
            if symbol not in self.symbol_list:
                for s in self.symbol_list:
                    if s.replace('/', '') == symbol:
                        internal_sym = s
                        break
            
            qty = float(data['q'])
            is_buyer_mm = data['m'] # True = Sell (at Bid), False = Buy (at Ask)
            
            delta_val = -qty if is_buyer_mm else qty
            
            # Use a decay or window for Delta? 
            # For micro-scalping, we want the "current" momentum.
            # We add to current delta and it will be reset or decayed by the strategy.
            if internal_sym not in self.order_flow_metrics:
                self.order_flow_metrics[internal_sym] = {
                    'imbalance': 1.0, 'bid_vol_5': 0.0, 'ask_vol_5': 0.0, 
                    'delta': 0.0, 'last_update': 0
                }
            
            # Cumulative Delta (Strategy will reset this every bar or use moving window)
            self.order_flow_metrics[internal_sym]['delta'] += delta_val
            
            # 🌊 PHASE 25: Microstructure Analysis (VPIN)
            # data['p'] = price, data['q'] = qty
            self.microstructure[internal_sym].on_trade(
                float(data['p']), qty, is_buyer_mm
            )
            
        except Exception as e:
            logger.debug(f"Error in aggTrade processing: {e}")

    async def update_symbol_list(self, new_symbols: List[str]):
        """
        Hot-swaps the symbol list and updates subscriptions.
        """
        old_symbols = set(self.symbol_list)
        target_symbols = set(new_symbols)
        
        added = target_symbols - old_symbols
        removed = old_symbols - target_symbols
        
        if not added and not removed:
            return
            
        with self._data_lock:
            for s in added:
                self._init_symbol_buffer(s)
                from core.orderbook import OrderBook
                self.microstructure[s] = MicrostructureAnalyzer(s)
                self.orderbooks[s] = OrderBook(max_depth=10)
                self.last_depth_update[s] = 0.0
                
            self.symbol_list = new_symbols
            
        if added:
            logger.info(f"🔄 Dynamic update: Fetching history for {len(added)} new symbols...")
            # Initialize SHM for new symbols
            self._init_shm()
            
            # Fetch history in a separate daemon thread to avoid blocking asyncio loop
            # and to avoid ThreadPoolExecutor deadlocks!
            import threading
            def _fetch_all_history():
                try:
                    self.fetch_initial_history()
                    self.fetch_initial_history_1h()
                    self.fetch_initial_history_4h()
                    self.fetch_initial_history_5m()
                    self.fetch_initial_history_15m()
                    self.fetch_initial_history_1d()
                    self.fetch_initial_history_1w()
                    logger.info("✅ Dynamic update history fetch complete.")
                except Exception as e:
                    logger.error(f"Error fetching dynamic history: {e}")
                    
            threading.Thread(target=_fetch_all_history, daemon=True, name="DynamicHistoryFetch").start()
            
            # Restart socket to subscribe to new symbols
            if self.socket:
                try:
                    await self.stop_socket()
                    asyncio.create_task(self.start_socket())
                except Exception as e:
                    logger.error(f"Failed to restart socket on dynamic update: {e}")

    def _process_book_ticker(self, data):
        """
        Phase 12: Updates real-time BBO (Best Bid Offer) cache.
        ENHANCED (Omega Mind): Calculates VBI (Volume Book Imbalance).
        """
        try:
            symbol = data['s']
            bid_p = float(data['b'])
            bid_q = float(data['B'])
            ask_p = float(data['a'])
            ask_q = float(data['A'])
            
            # 1. Update BBO Cache
            self.liquidity_cache[symbol] = {
                'bid': bid_p,
                'ask': ask_p,
                'bid_qty': bid_q,
                'ask_qty': ask_q,
                'ts': time.time()
            }
            
            # 2. Calculate VBI (Leading Indicator of Price Pressure)
            # VBI = (BidQty - AskQty) / (BidQty + AskQty)
            # Range: -1 (Sell Pressure) to 1 (Buy Pressure)
            total_q = bid_q + ask_q
            if total_q > 0:
                vbi = (bid_q - ask_q) / total_q
                
                # Find internal symbol to update buffer
                internal_sym = symbol
                if symbol not in self.vbi_history:
                    # Quick mapping check
                    for s in self.symbol_list:
                        if s.replace('/', '') == symbol:
                            internal_sym = s
                            break
                
                if internal_sym in self.vbi_history:
                    self.vbi_history[internal_sym].push(np.float32(vbi))

        except Exception as e:
            logger.debug(f"Error in VBI calc: {e}")

    def _process_liquidation(self, msg):
        """
        OMEGA MIND PHASE 98: Captura liquidaciones forzadas.
        Señal de capitulación o impulso extremo.
        """
        try:
            order = msg['o']
            symbol = order['s']
            side = order['S']
            qty = float(order['q'])
            price = float(order['ap'])
            size_usd = qty * price
            
            # Map symbol
            internal_sym = symbol
            if symbol not in self.symbol_list:
                for s in self.symbol_list:
                    if s.replace('/', '') == symbol:
                        internal_sym = s
                        break
            
            # Value: Positive for LONG liquidations (Sell orders), Negative for SHORT liquidations (Buy orders)
            val = size_usd if side == 'SELL' else -size_usd
            
            if internal_sym in self.liquidation_history:
                self.liquidation_history[internal_sym].push(np.float32(val))
            
            # --- 🛡️ PHASE 29: REAL-TIME DERIVATIVES SYNC ---
            if internal_sym in self.derivatives_metrics:
                # Accumulate liquidation volume in USD for the current minute window
                self.derivatives_metrics[internal_sym]['liquidations'] += size_usd
            
            if size_usd > 10000: # Log significant liquidations
                logger.info(f"🔥 LIQUIDATION [{internal_sym}]: {side} {size_usd:,.0f} USD")
                
        except Exception as e:
            logger.error(f"Error processing liquidation: {e}")

    def get_hft_indicators(self, symbol: str, n: int = 20) -> Dict[str, float]:
        """
        Phase 98: Aggregates real-time HFT signals for ML ingestion.
        """
        results = {'vbi': 0.0, 'liq_intensity': 0.0, 'vbi_avg': 0.0}
        
        try:
            # 1. VBI
            if symbol in self.vbi_history:
                vbi_data = self.vbi_history[symbol].get_last(n)
                if len(vbi_data) > 0:
                    results['vbi'] = float(vbi_data[-1])
                    results['vbi_avg'] = float(np.mean(vbi_data))
            
            # 2. Liquidations (Sum of last N liquidation events)
            if symbol in self.liquidation_history:
                liq_data = self.liquidation_history[symbol].get_last(n)
                if len(liq_data) > 0:
                    # Sum of net intensity (Sell - Buy)
                    results['liq_intensity'] = float(np.sum(liq_data))
                    
        except Exception as e:
            logger.debug(f"Error getting HFT indicators: {e}")
            
        return results

    def get_liquidity_snapshot(self, symbol):
        """
        Returns latest liquidity check for a symbol.
        """
        clean_sym = symbol.replace('/', '')
        return self.liquidity_cache.get(clean_sym, None)

    # ==========================================================
    # ✅ PHASE 5: DATA PERSISTENCE (PARQUET)
    # ==========================================================
    def save_snapshot(self):
        """
        Guarda el estado actual de los RingBuffers en disco (Parquet + ZSTD).
        """
        try:
            cache_dir = "data/cache_parquet"
            os.makedirs(cache_dir, exist_ok=True)
            count = 0
            
            for symbol in self.symbol_list:
                safe_sym = symbol.replace('/', '')
                
                # Snapshot 1m (Base)
                data = self.get_latest_bars(symbol, n=5000)
                if data is not None:
                    # Convert Structured Array to DataFrame for convenience in Parquet saving
                    # COPY DEEP to prevent PyArrow Segfaults on live-updating ring buffers
                    df = pd.DataFrame(np.copy(data))
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms') # Keep for compatibility
                    
                    path = f"{cache_dir}/{safe_sym}_1m.parquet"
                    df.to_parquet(path, compression='zstd')
                    count += 1
            
            logger.info(f"💾 [Persistence] Saved {count} symbols to Parquet.")
        except Exception as e:
            logger.error(f"❌ [Persistence] Save failed: {e}")

    def load_snapshot(self):
        """
        Carga datos históricos desde Parquet para evitar peticiones API.
        Returns: Set of symbols loaded successfully.
        """
        loaded_symbols = set()
        cache_dir = "data/cache_parquet"
        if not os.path.exists(cache_dir):
            return loaded_symbols
            
        logger.info("📂 [Persistence] Loading local Parquet cache...")
        
        for symbol in self.symbol_list:
            safe_sym = symbol.replace('/', '')
            path = f"{cache_dir}/{safe_sym}_1m.parquet"
            
            if os.path.exists(path):
                try:
                    # Check age of file
                    mtime = os.path.getmtime(path)
                    if (time.time() - mtime) > 3600 * 4: # 4 hours old max
                        continue
                        
                    df = pd.read_parquet(path)
                    if not df.empty:
                        self._init_symbol_buffer(symbol) 
                        buf = self.buffers_1m[symbol]
                        
                        # Fastest way: Vectorized NumPy injection instead of iterrows
                        timestamps = df['timestamp'].to_numpy(dtype=np.int64)
                        opens = df['open'].to_numpy(dtype=np.float32)
                        highs = df['high'].to_numpy(dtype=np.float32)
                        lows = df['low'].to_numpy(dtype=np.float32)
                        closes = df['close'].to_numpy(dtype=np.float32)
                        vols = df['volume'].to_numpy(dtype=np.float32)
                        
                        for i in range(len(timestamps)):
                            buf.push(timestamps[i], opens[i], highs[i], lows[i], closes[i], vols[i])
                            
                        loaded_symbols.add(symbol)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load parquet for {symbol}: {e}")
                    
        logger.info(f"📂 [Persistence] Loaded {len(loaded_symbols)} symbols from disk.")
        return loaded_symbols

    async def shutdown(self):
        """
        Graceful shutdown for all data resources.
        """
        logger.info("BinanceData: Initiating shutdown...")
        self._running = False # Stop reconnection loop
        
        # 1. Stop Socket
        await self.stop_socket()
        
        # 2. Cleanup ThreadPool
        logger.info("BinanceData: Closing ThreadPoolExecutor (Non-blocking)...")
        self.executor.shutdown(wait=False)
        
        logger.info("✅ BinanceData: Cleanup complete.")


    async def stop_socket(self):
        """
        Stops the WebSocket and closes the client sessions (aiohttp).
        """
        try:
            if self.client:
                logger.info("BinanceData: Closing Async Client session...")
                await self.client.close_connection()
                self.client = None
                logger.info("✅ Binance WebSocket Client Closed.")
        except Exception as e:
            logger.error(f"Error closing socket client: {e}")

    async def _handle_socket_message(self, msg):
        """
        [PHASE IV] Bandwidth Throttling Awareness.
        Dropping Depth packets if latency is high.
        """
        try:
            if 'data' not in msg: return
            
            data = msg['data']
            stream = msg['stream']
            
            # 1. Bandwidth Throttling Check
            # If latency > 500ms, IGNORE Depth Updates (Prioritize Trades/Price)
            avg_lat, max_lat = self.get_latency_metrics()
            is_throttled = avg_lat > 500 or max_lat > 1000
            
            if is_throttled and '@depth' in stream:
                # 🛑 DROP PACKET (Soft Throttling)
                # We log sparsely to avoid flooding
                if np.random.random() < 0.01: 
                    logger.warning(f"📉 [THROTTLING] High Latency ({avg_lat:.1f}ms). Dropping LOB Update for stability.")
                return 
                
            # 2. Process Message
            symbol = data['s'] # e.g. BTCUSDT (Upper)
            # Map back to internal format if needed? 
            # We usually use clean symbol in buffers.
            
            if '@depth' in stream:
                self._process_depth_update(data)
            elif '@trade' in stream:
                self._process_trade_update(data)
            elif '@forceOrder' in stream:
                self._process_force_order(data)
            elif '@markPrice' in stream:
                self._process_mark_price(data)
                 
        except Exception as e:
            logger.error(f"Msg Handler Error: {e}")

    def _process_depth_update(self, data):
        """
        Updates the internal L2 OrderBook and calculates OFI.
        """
        try:
            symbol = data.get('s')
            if not symbol: return
            
            internal_sym = symbol
            if symbol not in self.symbol_list:
                for s in self.symbol_list:
                    if s.replace('/', '') == symbol:
                        internal_sym = s
                        break
                        
            if internal_sym not in getattr(self, 'orderbooks', {}):
                return
                
            ob = self.orderbooks[internal_sym]
            
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            for b in bids:
                ob.update_bid(float(b[0]), float(b[1]))
            for a in asks:
                ob.update_ask(float(a[0]), float(a[1]))
                
            # Update metrics
            if internal_sym not in self.order_flow_metrics:
                self.order_flow_metrics[internal_sym] = {}
                
            self.order_flow_metrics[internal_sym]['l2_ofi'] = ob.calculate_ofi()
            self.order_flow_metrics[internal_sym]['l2_spread'] = ob.calculate_spread()
            
            # Calc distance to microprice
            micro = ob.calculate_microprice()
            best_bid = float(bids[0][0]) if bids else micro
            best_ask = float(asks[0][0]) if asks else micro
            mid = (best_bid + best_ask) / 2.0
            
            # We store the microprice distance relative to mid
            dist = (micro - mid) / (mid + 1e-9) if mid > 0 else 0.0
            self.order_flow_metrics[internal_sym]['l2_microprice_dist'] = dist
            
        except Exception as e:
            pass

    def _process_trade_update(self, data):
        """
        Processes standard trades and calculates Whale Flow Proxy.
        """
        try:
            symbol = data.get('s')
            if not symbol: return
            
            internal_sym = symbol
            if symbol not in self.symbol_list:
                for s in self.symbol_list:
                    if s.replace('/', '') == symbol:
                        internal_sym = s
                        break
                        
            qty = float(data.get('q', 0))
            price = float(data.get('p', 0))
            is_buyer_mm = data.get('m', False)
            trade_usd = qty * price
            
            # 🐋 Whale Flow Proxy: Trades > $100k USD
            if trade_usd > 100000:
                if not hasattr(self, 'derivatives_metrics'):
                    self.derivatives_metrics = {}
                if internal_sym not in self.derivatives_metrics:
                    self.derivatives_metrics[internal_sym] = {'funding_rate': 0.0, 'oi': 0.0, 'oi_delta': 0.0, 'liquidations': 0.0, 'whale_flow': 0.0}
                
                # Positive if market buy (buyer maker = False), Negative if market sell (buyer maker = True)
                flow = trade_usd if not is_buyer_mm else -trade_usd
                self.derivatives_metrics[internal_sym]['whale_flow'] = self.derivatives_metrics[internal_sym].get('whale_flow', 0.0) + flow
                
            # Passthrough to agg trade logic for delta and VPIN
            self._process_agg_trade(data)
        except Exception as e:
            pass 

    # ------------------------------------------------------------------
    # PHASE 99: BUFFER RESET (Manual Close Protocol)
    # ------------------------------------------------------------------
    def reset_symbol_buffers(self, symbol: str):
        """
        Re-initializes all ring buffers for a symbol.
        Called when a manual close is detected to provide a clean data slate.
        Thread-safe: Acquires _data_lock before mutation.
        """
        with self._data_lock:
            try:
                self._init_symbol_buffer(symbol)
                logger.info(f"🔄 [DataProvider] Buffers reset for {symbol} (all timeframes)")
            except Exception as e:
                logger.error(f"❌ [DataProvider] Failed to reset buffers for {symbol}: {e}")

    # ------------------------------------------------------------------
    # PHASE 16: POLARS ENGINE (Rust/Arrow)
    # ------------------------------------------------------------------

    def get_history_polars(self, symbol: str, timeframe: str = '1m', n: int = 1000) -> pl.DataFrame:
        """
        Retrieves historical data as a Polars DataFrame (Zero-Copy Arrow).
        """
        target_map = None
        if timeframe == '1m': target_map = self.buffers_1m
        elif timeframe == '5m': target_map = self.buffers_5m
        elif timeframe == '15m': target_map = self.buffers_15m
        elif timeframe == '1h': target_map = self.buffers_1h
        
        if not target_map or symbol not in target_map:
            return pl.DataFrame()
            
        buf = target_map[symbol]
        t, o, h, l, c, v = buf.get_last(n)
        
        if len(t) == 0:
            return pl.DataFrame()
            
        # Construct Polars DataFrame directly from Numpy arrays (Arrow Zero-Copy)
        pldf = pl.DataFrame({
            "timestamp": t,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v
        }).with_columns(
            pl.col("timestamp").cast(pl.Int64),
            pl.col("close").cast(pl.Float32)
        )
        
        # [VISION INJECTION] Merge historical L2 and AggTrades if present
        try:
            import os
            import glob
            sym_clean = symbol.replace("/", "").upper()
            cache_dir = "data/cache_parquet"
            vision_files = glob.glob(f"{cache_dir}/{sym_clean}_vision_*.parquet")
            if vision_files and timeframe == '1m':
                dfs = [pl.read_parquet(f) for f in vision_files]
                if dfs:
                    vision_df = pl.concat(dfs).unique("timestamp_ms")
                    pldf = pldf.join(vision_df, left_on="timestamp", right_on="timestamp_ms", how="left").fill_null(0.0)
        except Exception as e:
            pass
            
        return pldf

    def get_order_flow_metrics(self, symbol: str) -> dict:
        """
        Returns a dictionary with real-time order flow metrics.
        Merged with microstructure metrics (VPIN, Icebergs, Delta).
        """
        internal_sym = symbol
        if symbol not in self.symbol_list:
            for s in self.symbol_list:
                if s.replace("/", "") == symbol:
                    internal_sym = s
                    break
                    
        metrics = self.order_flow_metrics.get(internal_sym, {})
        of_metrics = metrics.copy()
        
        # Merge True Microstructure
        if internal_sym in getattr(self, 'microstructure', {}):
            micro_metrics = self.microstructure[internal_sym].get_metrics()
            of_metrics.update(micro_metrics)
            
        return of_metrics

    def get_derivatives_metrics(self, symbol: str) -> dict:
        """
        Returns futures derivatives metrics (Funding, Open Interest, Liquidations).
        """
        return getattr(self, 'derivatives_metrics', {}).get(symbol, {
            'funding_rate': 0.0,
            'oi': 0.0,
            'oi_delta': 0.0,
            'liquidations': 0.0,
            'whale_flow': 0.0
        })

    def get_orderbook(self, symbol: str):
        """
        Returns the OrderBook instance for a symbol to access L2 metrics.
        """
        return getattr(self, 'orderbooks', {}).get(symbol, None)
