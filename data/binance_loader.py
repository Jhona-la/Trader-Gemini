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


class BinanceData(DataProvider):
    def __init__(self, events_queue, symbol_list):
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        self._running = True
        self.client_sync = None
        
        # 1. Thread Pool for Parallel Fetching (I/O Bound)
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="BinanceFetch")
        
        # 2. Data Buffers Dictionary Initialization (Phase 9/98)
        self.buffers_1m = {}
        self.buffers_5m = {}
        self.buffers_15m = {}
        self.buffers_1h = {}
        self.vbi_history = {}
        self.liquidation_history = {}
        self.last_event_time = {}
        self.liquidity_cache = {}
        # PHASE 13: Order Flow Metrics
        self.order_flow_metrics = {}
        
        # PHASE 14: Lead-Lag Intelligence
        self.lead_lag_results = {} # {symbol: lag_in_seconds}
        self.reference_symbol = "BTC/USDT"
        
        # 🌊 PHASE 25: Microstructure Analyzers
        self.microstructure = {} 
        for s in self.symbol_list:
            self.microstructure[s] = MicrostructureAnalyzer(s)


        # 3. Init actual Numba buffers for each symbol
        for s in self.symbol_list:
            self._init_symbol_buffer(s)
            
        # 4. Thread Safety Lock
        import threading
        self._data_lock = threading.Lock()

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
                    # Unlink if exists (Cleanup)
                    try:
                        existing = shared_memory.SharedMemory(name=name)
                        existing.close()
                        existing.unlink()
                    except: pass
                    
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
             
        self.client_sync = Client(
            api_key, 
            api_secret, 
            testnet=Config.BINANCE_USE_TESTNET,
            requests_params={'timeout': 20}
        )
        
        # Phase 38: Keep-Alive Tuning
        from utils.keep_alive import tune_requests_session
        tune_requests_session(self.client_sync.session)
        
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
        self.fetch_initial_history_5m()  # NEW
        self.fetch_initial_history_15m()  # NEW
        
        # Async Client & Socket Manager placeholders
        self.client = None
        self.bsm = None
        self.socket = None
        
        # Phase 16: Latency Circuit Breaker Stats
        self.latency_history = collections.deque(maxlen=20)
        self._start_latency_monitor()

    def _start_latency_monitor(self):
        """Starts a background thread to ping Binance every 5s."""
        def _ping_loop():
            while self._running:
                try:
                    t0 = time.time()
                    self.client_sync.ping()
                    t1 = time.time()
                    latency_ms = (t1 - t0) * 1000
                    self.latency_history.append(latency_ms)
                except Exception as e:
                    logger.error(f"Ping failed: {e}")
                    self.latency_history.append(9999.0) # Penalty for timeout
                
                time.sleep(5) # Check every 5s

        import threading
        t = threading.Thread(target=_ping_loop, daemon=True, name="LatencyMonitor")
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
            
            return results
        except Exception as e:
            logger.error(f"Fetch failed for {s}: {e}")
            return None

    # ... (fetch methods remain same, but we should lock inside them if they were called concurrently, 
    # but they are called in init so it's fine. The critical part is update_bars vs get_latest_bars)

    def get_latest_bars(self, symbol, n=1, timeframe='1m'):
        """
        Returns NumPy Structured Array (Phase 4 Zero-Copy).
        Represents OHLCVT data without Pandas overhead.
        """
        try:
            with self._data_lock:
                target_map = self.buffers_1m
                if timeframe == '5m': target_map = self.buffers_5m
                elif timeframe == '15m': target_map = self.buffers_15m
                elif timeframe == '1h': target_map = self.buffers_1h
                
                if symbol not in target_map:
                    return None
                
                buf = target_map[symbol]
                if buf.size == 0: return None
                
                # Retrieve from Numba (as tuple of arrays)
                t, o, h, l, c, v = buf.get_last(n)
                
                # Convert to Structured Array
                # Dtype definition matches SUPREMO-V3 standard
                struct_dtype = [
                    ('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'), 
                    ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')
                ]
                
                # Optimized construction
                res = np.empty(len(t), dtype=struct_dtype)
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


    def _fetch_deep_history_worker(self, symbol, interval, hours, dest_dict, limit_per_req=1000, buffer_multiplier=1.2):
        """Worker function for parallel history fetching"""
        try:
            # Calculate requirements
            time_needed = Config.Strategies.ML_LOOKBACK_BARS * buffer_multiplier if interval == '1m' else hours * 60
            minutes_needed = hours * 60
            total_candles_needed = minutes_needed / (15 if interval == '15m' else 60 if interval == '1h' else 5 if interval == '5m' else 1)
            
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
            
            sym_clean = symbol.replace('/', '')
            
            while len(all_candles) < total_candles_needed:
                candles = self.client_sync.get_klines(symbol=sym_clean, interval=kl_interval, limit=limit_per_req, startTime=since)
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + (60000 * (15 if interval == '15m' else 60 if interval == '1h' else 5 if interval == '5m' else 1))
                
                if len(candles) < limit_per_req:
                    break
            
            # Process and store
            processed_bars = []
            for c in all_candles:
                # No pd.to_datetime here (Phase 1: Zero-Pandas)
                processed_bars.append({
                    'symbol': symbol,
                    'timestamp': int(c[0]),
                    'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])
                })
            
            # RAM OPTIMIZATION (Filled into Ring Buffers)
            if interval == '1m': target_map = self.buffers_1m
            elif interval == '5m': target_map = self.buffers_5m
            elif interval == '15m': target_map = self.buffers_15m
            elif interval == '1h': target_map = self.buffers_1h
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
        for f in as_completed(futures):
            pass # exceptions logged in worker

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




    def get_latest_bars_1h(self, symbol, n=1):
        """
        Returns the last N 1h bars (RingBuffer Wrapper).
        """
        return self.get_latest_bars(symbol, n, timeframe='1h')

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
                    with self._data_lock:
                         buf = self.buffers_1m[s]
                         last_t_arr = buf.get_last(1)
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
                    
                # --- PHASE 14: LEAD-LAG SYNC ---
                if s != self.reference_symbol and self.reference_symbol in self.buffers_1m:
                    self._calculate_lead_lag(s)
                    
            except Exception as e:
                logger.error(f"Error processing update for {s}: {e}")
                
                # MULTI-TIMEFRAME: Also update 1h candles
                # Fetch 1h candle (limit=2 to get latest closed or current open)
                # bars_1h = self.exchange.fetch_ohlcv(s, timeframe='1h', limit=2)
                bars_1h = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1HOUR, limit=2)
                if bars_1h:
                    latest_1h = bars_1h[-1]
                    ts_1h = pd.to_datetime(latest_1h[0], unit='ms')
                    
                    # Update if new or same (to update current candle)
                    # For trend analysis, we usually want closed candles, but updating current is fine
                    bar_data_1h = {
                        'symbol': s,
                        'datetime': ts_1h,
                        'open': float(latest_1h[1]),
                        'high': float(latest_1h[2]),
                        'low': float(latest_1h[3]),
                        'close': float(latest_1h[4]),
                        'volume': float(latest_1h[5])
                    }
                    
                    if not self.buffers_1h.get(s): return

                    # Push to 1H Ring Buffer
                    buf = self.buffers_1h[s]
                    # Check if new or update last
                    ts_1h_ms = int(ts_1h.timestamp() * 1000)
                    last_arr = buf['t'].get_last(1)
                    
                    if len(last_arr) > 0 and last_arr[0] == ts_1h_ms:
                         # Overwrite logic (manual head rewind)
                         buf['t'].head = (buf['t'].head - 1 + buf['t'].capacity) % buf['t'].capacity
                         buf['o'].head = (buf['o'].head - 1 + buf['o'].capacity) % buf['o'].capacity
                         buf['h'].head = (buf['h'].head - 1 + buf['h'].capacity) % buf['h'].capacity
                         buf['l'].head = (buf['l'].head - 1 + buf['l'].capacity) % buf['l'].capacity
                         buf['c'].head = (buf['c'].head - 1 + buf['c'].capacity) % buf['c'].capacity
                         buf['v'].head = (buf['v'].head - 1 + buf['v'].capacity) % buf['v'].capacity
                         
                         if buf['t'].size > 0: buf['t'].size -= 1
                         if buf['o'].size > 0: buf['o'].size -= 1
                         if buf['h'].size > 0: buf['h'].size -= 1
                         if buf['l'].size > 0: buf['l'].size -= 1
                         if buf['c'].size > 0: buf['c'].size -= 1
                         if buf['v'].size > 0: buf['v'].size -= 1

                    buf['t'].push(ts_1h_ms)
                    buf['o'].push(np.float32(latest_1h[1]))
                    buf['h'].push(np.float32(latest_1h[2]))
                    buf['l'].push(np.float32(latest_1h[3]))
                    buf['c'].push(np.float32(latest_1h[4]))
                    buf['v'].push(np.float32(latest_1h[5]))
                
                # MULTI-TIMEFRAME: Also update 5m candles
                # bars_5m = self.exchange.fetch_ohlcv(s, timeframe='5m', limit=2)
                bars_5m = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_5MINUTE, limit=2)
                if bars_5m:
                    latest_5m = bars_5m[-1]
                    ts_5m = pd.to_datetime(latest_5m[0], unit='ms')
                    
                    bar_data_5m = {
                        'symbol': s,
                        'datetime': ts_5m,
                        'open': float(latest_5m[1]),
                        'high': float(latest_5m[2]),
                        'low': float(latest_5m[3]),
                        'close': float(latest_5m[4]),
                        'volume': float(latest_5m[5])
                    }
                    
                    # Update 5m Ring Buffer
                    if s in self.buffers_5m:
                        buf = self.buffers_5m[s]
                        ts_5m_ms = int(ts_5m.timestamp() * 1000)
                        last_arr = buf['t'].get_last(1)
                        if len(last_arr) > 0 and last_arr[0] == ts_5m_ms:
                             buf['t'].head = (buf['t'].head - 1 + buf['t'].capacity) % buf['t'].capacity
                             buf['o'].head = (buf['o'].head - 1 + buf['o'].capacity) % buf['o'].capacity
                             buf['h'].head = (buf['h'].head - 1 + buf['h'].capacity) % buf['h'].capacity
                             buf['l'].head = (buf['l'].head - 1 + buf['l'].capacity) % buf['l'].capacity
                             buf['c'].head = (buf['c'].head - 1 + buf['c'].capacity) % buf['c'].capacity
                             buf['v'].head = (buf['v'].head - 1 + buf['v'].capacity) % buf['v'].capacity
                             if buf['t'].size > 0: buf['t'].size -= 1
                             if buf['o'].size > 0: buf['o'].size -= 1
                             if buf['h'].size > 0: buf['h'].size -= 1
                             if buf['l'].size > 0: buf['l'].size -= 1
                             if buf['c'].size > 0: buf['c'].size -= 1
                             if buf['v'].size > 0: buf['v'].size -= 1

                        buf['t'].push(ts_5m_ms)
                        buf['o'].push(np.float32(latest_5m[1]))
                        buf['h'].push(np.float32(latest_5m[2]))
                        buf['l'].push(np.float32(latest_5m[3]))
                        buf['c'].push(np.float32(latest_5m[4]))
                        buf['v'].push(np.float32(latest_5m[5]))
                
                # MULTI-TIMEFRAME: Also update 15m candles
                # bars_15m = self.exchange.fetch_ohlcv(s, timeframe='15m', limit=2)
                bars_15m = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_15MINUTE, limit=2)
                if bars_15m:
                    latest_15m = bars_15m[-1]
                    ts_15m = pd.to_datetime(latest_15m[0], unit='ms')
                    
                    bar_data_15m = {
                        'symbol': s,
                        'datetime': ts_15m,
                        'open': float(latest_15m[1]),
                        'high': float(latest_15m[2]),
                        'low': float(latest_15m[3]),
                        'close': float(latest_15m[4]),
                        'volume': float(latest_15m[5])
                    }
                    
                    # Update 15m Ring Buffer
                    if s in self.buffers_15m:
                        buf = self.buffers_15m[s]
                        ts_15m_ms = int(ts_15m.timestamp() * 1000)
                        last_arr = buf['t'].get_last(1)
                        if len(last_arr) > 0 and last_arr[0] == ts_15m_ms:
                             buf['t'].head = (buf['t'].head - 1 + buf['t'].capacity) % buf['t'].capacity
                             buf['o'].head = (buf['o'].head - 1 + buf['o'].capacity) % buf['o'].capacity
                             buf['h'].head = (buf['h'].head - 1 + buf['h'].capacity) % buf['h'].capacity
                             buf['l'].head = (buf['l'].head - 1 + buf['l'].capacity) % buf['l'].capacity
                             buf['c'].head = (buf['c'].head - 1 + buf['c'].capacity) % buf['c'].capacity
                             buf['v'].head = (buf['v'].head - 1 + buf['v'].capacity) % buf['v'].capacity
                             if buf['t'].size > 0: buf['t'].size -= 1
                             if buf['o'].size > 0: buf['o'].size -= 1
                             if buf['h'].size > 0: buf['h'].size -= 1
                             if buf['l'].size > 0: buf['l'].size -= 1
                             if buf['c'].size > 0: buf['c'].size -= 1
                             if buf['v'].size > 0: buf['v'].size -= 1

                        buf['t'].push(ts_15m_ms)
                        buf['o'].push(np.float32(latest_15m[1]))
                        buf['h'].push(np.float32(latest_15m[2]))
                        buf['l'].push(np.float32(latest_15m[3]))
                        buf['c'].push(np.float32(latest_15m[4]))
                        buf['v'].push(np.float32(latest_15m[5]))
                    
            except Exception as e:
                logger.error(f"Error calculating Lead-Lag for {symbol}: {e}")
                logger.error(f"Error fetching data for {s}: {e}")

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
            
            for lag in range(-5, 6):
                if lag == 0:
                    corr = np.corrcoef(ref_rets, target_rets)[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(ref_rets[lag:], target_rets[:-lag])[0, 1]
                else:
                    abs_lag = abs(lag)
                    corr = np.corrcoef(ref_rets[:-abs_lag], target_rets[abs_lag:])[0, 1]
                
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
        
        # Try alternate formats?
        return None


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
                        self._process_book_ticker(data)
                    elif 'forceOrder' in stream_name:
                        self._process_liquidation(data)
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
                
    async def process_socket_message(self, msg):
        """
        DEPRECATED: Logic moved to _manage_socket_chunk routing.
        Kept for compatibility if called directly, but essentially purely abstract now.
        """
        pass

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
            
            # Extract data with Phase 4: Downcasting (float32 for memory)
            timestamp = pd.to_datetime(kline['t'], unit='ms')
            open_price = np.float32(kline['o'])
            high_price = np.float32(kline['h'])
            low_price = np.float32(kline['l'])
            close_price = np.float32(kline['c'])
            volume = np.float32(kline['v'])

            # DATA QUALITY FILTER (Rule 3.2 - Refined)
            if not is_closed:
                if volume <= 0 or (high_price == low_price):
                    return
            elif volume <= 0:
                logger.debug(f"🕳️ [Loader] Liquidity Hole in {internal_symbol}: Zero volume bar recorded for time continuity.")

            # GAP DETECTION (Rule 3.2)
            with self._data_lock:
                if self.latest_data[internal_symbol]:
                    last_ts = self.latest_data[internal_symbol][-1]['datetime']
                    time_diff = (timestamp - last_ts).total_seconds()
                    
                    if time_diff > 65 and is_closed:
                        logger.warning(f"🚨 GAP DETECTED in {internal_symbol}: {time_diff}s interval. Data might be missing.")
            
            # ─── BUFFER UPDATE (Thread-Safe) ───
            # PHASE 33: Multi-Timeframe Routing via stream_name parameter
            tf = '1m'
            if '@kline_5m' in stream_name: tf = '5m'
            elif '@kline_15m' in stream_name: tf = '15m'
            elif '@kline_1h' in stream_name: tf = '1h'
            
            target_map = self.buffers_1m
            if tf == '5m': target_map = self.buffers_5m
            elif tf == '15m': target_map = self.buffers_15m
            elif tf == '1h': target_map = self.buffers_1h
            
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

            # Time-based throttle: at most once every 2s per symbol
            if not should_trigger:
                last_t = self.last_event_time.get(internal_symbol, 0)
                if now_ts - last_t > 2.0:
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
                        order_flow=of_metrics
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
                    ))
            
            if is_closed:
                logger.info(f"🌊 WebSocket Closed Bar: {internal_symbol} @ {close_price}")
                
        except Exception as e:
            logger.error(f"WebSocket Message Error: {e}")

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
            
            # Sum volume of top 5 levels
            bids = data['b'] # [[price, qty], ...]
            asks = data['a']
            
            bid_vol_5 = sum(float(b[1]) for b in bids[:5])
            ask_vol_5 = sum(float(a[1]) for a in asks[:5])
            
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
                    df = pd.DataFrame(data)
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
                        # Feed the buffer
                        self._init_symbol_buffer(symbol) 
                        # Use new push logic
                        buf = self.buffers_1m[symbol]
                        for _, row in df.iterrows():
                            buf.push(
                                int(row['timestamp']),
                                np.float32(row['open']),
                                np.float32(row['high']),
                                np.float32(row['low']),
                                np.float32(row['close']),
                                np.float32(row['volume'])
                            )
                            
                        loaded_symbols.add(symbol)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load parquet for {symbol}: {e}")
                    
        logger.info(f"📂 [Persistence] Loaded {len(loaded_symbols)} symbols from disk.")
        return loaded_symbols

    async def shutdown(self):
        """
        Graceful shutdown for all data resources.
        """
        self.save_snapshot() # Auto-save on exit
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
    async def start_socket(self):
        """
        [PHASE 99] WebSocket Implementation with Bandwidth Throttling (VANGUARDIA-SOBERANA).
        """
        try:
            if not self.client:
                # Initialize Async Client
                api_key = Config.BINANCE_API_KEY
                api_secret = Config.BINANCE_SECRET_KEY
                
                # Determine Environment
                testnet = Config.BINANCE_USE_TESTNET
                if testnet:
                    logger.info("🔌 [Socket] Connecting to TESTNET...")
                else:
                    logger.info("🔌 [Socket] Connecting to LIVE...")

                self.client = await AsyncClient.create(api_key, api_secret, testnet=testnet)
                self.bsm = BinanceSocketManager(self.client)
            
            # Construct Streams
            # We want: <symbol>@depth5@100ms and <symbol>@trade
            # Note: For HFT we might want @bookTicker for speed, but user asked for LOB depth scaling.
            # So standard is depth5.
            streams = []
            for s in self.symbol_list:
                clean = s.replace('/', '').lower()
                streams.append(f"{clean}@depth5@100ms")
                streams.append(f"{clean}@trade")
                
            logger.info(f"🔌 [Socket] Subscribing to {len(streams)} streams...")
            
            # Start Multiplex Socket
            self.socket = self.bsm.multiplex_socket(streams)
            
            # Start Event Loop
            async with self.socket as tscm:
                while self._running:
                    try:
                        msg = await tscm.recv()
                        if msg:
                            await self._handle_socket_message(msg)
                    except Exception as e:
                        logger.error(f"❌ Socket Error: {e}")
                        # Reconnect logic should be handled by outer loop or supervisor
                        await asyncio.sleep(5)
                        
        except Exception as e:
            logger.critical(f"❌ Critical Socket Failure: {e}")

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
                 
        except Exception as e:
            logger.error(f"Msg Handler Error: {e}")

    def _process_depth_update(self, data):
        # Implementation of pushing Bids/Asks to SHM or Buffers
        # For now, we update the LOB snapshot in memory
        pass

    def _process_trade_update(self, data):
        # Update price, volume buffers
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
        return pl.DataFrame({
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
