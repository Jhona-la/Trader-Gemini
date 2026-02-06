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
from collections import deque
try:
    import ujson as json
except ImportError:
    import json

class BinanceData(DataProvider):
    def __init__(self, events_queue, symbol_list):
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        self._running = True
        
        # Thread Pool for Parallel Fetching (I/O Bound)
        # Initialize BEFORE calling fetch methods
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="BinanceFetch")
        
        # Initialize python-binance Sync Client for REST calls
        # This replaces CCXT for all historical data fetching
        self.client_sync = None
        self._init_sync_client()
        
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
        
        # Storage for latest bars
        # OPTIMIZATION: Use deque with maxlen for O(1) appends and auto-discard
        MAX_1M = 2000
        MAX_5M = 500
        MAX_15M = 500
        MAX_1H = 500
        
        self.latest_data = {s: deque(maxlen=MAX_1M) for s in self.symbol_list}
        self.latest_data_1h = {s: deque(maxlen=MAX_1H) for s in self.symbol_list}  # 1h candles storage
        self.latest_data_5m = {s: deque(maxlen=MAX_5M) for s in self.symbol_list}  # NEW: 5m candles storage
        self.latest_data_15m = {s: deque(maxlen=MAX_15M) for s in self.symbol_list}  # NEW: 15m candles storage
        
        # Thread Safety Lock
        import threading
        self._data_lock = threading.Lock()
        
        # Throttling Tracking (NEW: Fixed missing attribute)
        self.last_event_time = {}
        
        # Fetch initial history at startup
        self.fetch_initial_history()
        self.fetch_initial_history_1h()
        self.fetch_initial_history_5m()  # NEW
        self.fetch_initial_history_15m()  # NEW
        
        # Async Client & Socket Manager placeholders
        self.client = None
        self.bsm = None
        self.socket = None
        # Thread Pool for Parallel Fetching (I/O Bound)
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="BinanceFetch")
        
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
                    'datetime': pd.to_datetime(latest[0], unit='ms'),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
            
            # Fetch 5m
            k5m = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_5MINUTE, limit=2)
            if k5m:
                latest = k5m[-1]
                results['5m'] = {
                    'datetime': pd.to_datetime(latest[0], unit='ms'),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }

            # Fetch 15m
            k15m = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_15MINUTE, limit=2)
            if k15m:
                latest = k15m[-1]
                results['15m'] = {
                    'datetime': pd.to_datetime(latest[0], unit='ms'),
                    'open': float(latest[1]), 'high': float(latest[2]), 'low': float(latest[3]),
                    'close': float(latest[4]), 'volume': float(latest[5])
                }
                
            # Fetch 1h
            k1h = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1HOUR, limit=2)
            if k1h:
                latest = k1h[-1]
                results['1h'] = {
                    'datetime': pd.to_datetime(latest[0], unit='ms'),
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
        Returns the last N bars from the requested timeframe.
        Thread-safe deep copy.
        """
        try:
            with self._data_lock:
                # Select target dictionary based on timeframe
                target_dict = self.latest_data
                if timeframe == '5m': target_dict = self.latest_data_5m
                elif timeframe == '15m': target_dict = self.latest_data_15m
                elif timeframe == '1h': target_dict = self.latest_data_1h
                
                if symbol not in target_dict:
                    return []
                    
                dq = target_dict[symbol]
                if not dq:
                    return []
                    
                if len(dq) < n:
                    return list(dq)
                
                return list(dq)[-n:]
        except Exception as e:
            logger.error(f"Error getting {timeframe} bars for {symbol}: {e}")
            return []


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
                dt = pd.to_datetime(c[0], unit='ms')
                processed_bars.append({
                    'symbol': symbol,
                    'datetime': dt,
                    'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])
                })
            
            # RAM OPTIMIZATION
            if interval == '1m':
                keep_size = max(Config.Strategies.ML_LOOKBACK_BARS + 500, 2000)
                dest_dict[symbol] = processed_bars[-keep_size:]
            else:
                dest_dict[symbol] = processed_bars
            
            logger.info(f"Loaded {len(dest_dict[symbol])} {interval} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to fetch {interval} history for {symbol}: {e}")

    def fetch_initial_history(self):
        """Fetches ~25 hours of historical data (1m candles) in PARALLEL."""
        time_needed = Config.Strategies.ML_LOOKBACK_BARS * 1.2
        hours = int(time_needed / 60) + 5
        logger.info(f"Fetching 1m historical data ({hours}h) in PARALLEL...")
        
        futures = []
        for s in self.symbol_list:
            futures.append(self.executor.submit(
                self._fetch_deep_history_worker, s, '1m', hours, self.latest_data
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
                self._fetch_deep_history_worker, s, '1h', hours, self.latest_data_1h
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
                self._fetch_deep_history_worker, s, '5m', hours, self.latest_data_5m
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
                self._fetch_deep_history_worker, s, '15m', hours, self.latest_data_15m
            ))
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            pass


    def get_latest_bars(self, symbol, n=1):
        """
        Returns the last N bars from the latest_symbol list.
        """
        try:
            # ZERO-TRUST: Return a DEEP COPY to prevent strategies from corrupting shared data
            import copy
            if symbol not in self.latest_data:
                return []
            bars_list = copy.deepcopy(self.latest_data[symbol])
        except KeyError:
            # logger.warning(f"Sort of expected: Symbol {symbol} not in dataset")
            return []
        except Exception as e:
            logger.error(f"Error getting bars: {e}")
            return []
        
        return bars_list[-n:]

    def get_latest_bars_1h(self, symbol, n=1):
        """
        Returns the last N 1h bars.
        """
        try:
            bars_list = self.latest_data_1h[symbol]
        except KeyError:
            logger.warning("That symbol is not available in the 1h data set.")
            raise
        
        return bars_list[-n:]

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
                    last_update_ms = int(data_packet['1m']['datetime'].timestamp() * 1000)
                    if (current_time_ms - last_update_ms) > 300000: # 5 minutes
                        logger.warning(f"⚠️ DATA STALE: {s} last update was {int((current_time_ms - last_update_ms)/60000)}m ago.")
                
                # 1. Update 1m Data (Primary)
                if '1m' in data_packet:
                    bar_data = data_packet['1m']
                    bar_data['symbol'] = s
                    timestamp = bar_data['datetime']
                    
                    is_new_bar = False
                    is_new_bar = False
                    with self._data_lock:
                        if not self.latest_data[s]:
                            self.latest_data[s].append(bar_data)
                            is_new_bar = True
                        elif self.latest_data[s][-1]['datetime'] != timestamp:
                            self.latest_data[s].append(bar_data)
                            # Deque handles maxlen automatically!
                            is_new_bar = True
                        else:
                            self.latest_data[s][-1] = bar_data
                            is_new_bar = False
                            
                    if is_new_bar:
                        logger.info(f"New Bar for {s}: {timestamp} - Close: {bar_data['close']}")
                        # OPTIMIZED: Enrich event with data to avoid lookup
                        self.events_queue.put(MarketEvent(
                            symbol=s, 
                            close_price=bar_data['close'],
                            timestamp=datetime.now()
                        ))

                # 2. Update 5m Data
                if '5m' in data_packet:
                    bar_data = data_packet['5m']
                    bar_data['symbol'] = s
                    timestamp = bar_data['datetime']
                    with self._data_lock:
                        if not self.latest_data_5m[s]:
                             self.latest_data_5m[s].append(bar_data)
                        elif self.latest_data_5m[s][-1]['datetime'] != timestamp:
                             self.latest_data_5m[s].append(bar_data)
                             # maxlen handled by deque
                        else:
                             self.latest_data_5m[s][-1] = bar_data

                # 3. Update 15m Data
                if '15m' in data_packet:
                    bar_data = data_packet['15m']
                    bar_data['symbol'] = s
                    timestamp = bar_data['datetime']
                    with self._data_lock:
                        if not self.latest_data_15m[s]:
                             self.latest_data_15m[s].append(bar_data)
                        elif self.latest_data_15m[s][-1]['datetime'] != timestamp:
                             self.latest_data_15m[s].append(bar_data)
                             if len(self.latest_data_15m[s]) > 200: self.latest_data_15m[s] = self.latest_data_15m[s][-200:]
                        else:
                             self.latest_data_15m[s][-1] = bar_data

                # 4. Update 1h Data
                if '1h' in data_packet:
                    bar_data = data_packet['1h']
                    bar_data['symbol'] = s
                    timestamp = bar_data['datetime']
                    with self._data_lock:
                        if not self.latest_data_1h[s]:
                             self.latest_data_1h[s].append(bar_data)
                        elif self.latest_data_1h[s][-1]['datetime'] != timestamp:
                             self.latest_data_1h[s].append(bar_data)
                             if len(self.latest_data_1h[s]) > 500: self.latest_data_1h[s] = self.latest_data_1h[s][-500:]
                        else:
                             self.latest_data_1h[s][-1] = bar_data
                    
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
                    
                    # Update storage (replace last if same time, append if new)
                    if not self.latest_data_1h[s]:
                        self.latest_data_1h[s].append(bar_data_1h)
                    elif self.latest_data_1h[s][-1]['datetime'] == ts_1h:
                        self.latest_data_1h[s][-1] = bar_data_1h
                    else:
                        self.latest_data_1h[s].append(bar_data_1h)
                        # Keep size manageable
                        if len(self.latest_data_1h[s]) > 500:
                            self.latest_data_1h[s] = self.latest_data_1h[s][-500:]
                
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
                    
                    # Update storage (replace last if same time, append if new)
                    if not self.latest_data_5m[s]:
                        self.latest_data_5m[s].append(bar_data_5m)
                    elif self.latest_data_5m[s][-1]['datetime'] == ts_5m:
                        self.latest_data_5m[s][-1] = bar_data_5m
                    else:
                        self.latest_data_5m[s].append(bar_data_5m)
                        if len(self.latest_data_5m[s]) > 200:
                            self.latest_data_5m[s] = self.latest_data_5m[s][-200:]
                
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
                    
                    # Update storage (replace last if same time, append if new)
                    if not self.latest_data_15m[s]:
                        self.latest_data_15m[s].append(bar_data_15m)
                    elif self.latest_data_15m[s][-1]['datetime'] == ts_15m:
                        self.latest_data_15m[s][-1] = bar_data_15m
                    else:
                        self.latest_data_15m[s].append(bar_data_15m)
                        if len(self.latest_data_15m[s]) > 150:
                            self.latest_data_15m[s] = self.latest_data_15m[s][-150:]
                    
            except Exception as e:
                logger.error(f"Error fetching data for {s}: {e}")

    def fetch_initial_history_5m(self):
        """
        Fetches 5-minute candles for multi-timeframe analysis.
        Loads last 150 bars (~12.5 hours) to support ML features.
        """
        logger.info("Fetching 5m historical data from Binance...")
        limit = 150  # Get 150 x 5m bars = 750 minutes = 12.5 hours
        
        for s in self.symbol_list:
            try:
                # candles = self.exchange.fetch_ohlcv(s, timeframe='5m', limit=limit)
                sym_clean = s.replace('/', '')
                candles = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_5MINUTE, limit=limit)
                
                processed_bars = []
                for c in candles:
                    timestamp = pd.to_datetime(c[0], unit='ms')
                    processed_bars.append({
                        'symbol': s,
                        'datetime': timestamp,
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    })
                
                self.latest_data_5m[s] = processed_bars
                logger.info(f"Loaded {len(self.latest_data_5m[s])} 5m bars for {s}")
                
                # Rate limit
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to fetch 5m history for {s}: {e}")
    
    def fetch_initial_history_15m(self):
        """
        Fetches 15-minute candles for multi-timeframe analysis.
        Loads last 100 bars (~25 hours) to support ML features.
        """
        logger.info("Fetching 15m historical data from Binance...")
        limit = 100  # Get 100 x 15m bars = 1500 minutes = 25 hours
        
        for s in self.symbol_list:
            try:
                # candles = self.exchange.fetch_ohlcv(s, timeframe='15m', limit=limit)
                sym_clean = s.replace('/', '')
                candles = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_15MINUTE, limit=limit)
                
                processed_bars = []
                for c in candles:
                    timestamp = pd.to_datetime(c[0], unit='ms')
                    processed_bars.append({
                        'symbol': s,
                        'datetime': timestamp,
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5])
                    })
                
                self.latest_data_15m[s] = processed_bars
                logger.info(f"Loaded {len(self.latest_data_15m[s])} 15m bars for {s}")
                
                # Rate limit
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to fetch 15m history for {s}: {e}")
    
    def get_latest_bars_5m(self, symbol, n=20):
        """
        Returns the last N 5-minute bars for the symbol.
        Used by ML Strategy for multi-timeframe analysis.
        """
        try:
            return self.latest_data_5m.get(symbol, [])[-n:]
        except:
            return []
    
    def get_latest_bars_15m(self, symbol, n=20):
        """
        Returns the last N 15-minute bars for the symbol.
        Used by ML Strategy for multi-timeframe analysis.
        """
        try:
            return self.latest_data_15m.get(symbol, [])[-n:]
        except:
            return []

    async def start_socket(self):
        """
        Starts the WebSocket connection for real-time data updates.
        """
        logger.info("Starting Binance WebSocket...")
        
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
        
        # Reconnection Loop
        while self._running:
            try:
                self.client = await AsyncClient.create(api_key, api_secret, testnet=Config.BINANCE_USE_TESTNET)
                self.bsm = BinanceSocketManager(self.client)
                
                # Create streams for all symbols
                streams = [f"{s.lower().replace('/', '')}@kline_1m" for s in self.symbol_list]
                
                logger.info(f"Subscribing to streams: {streams}")
                self.socket = self.bsm.multiplex_socket(streams)
                
                async with self.socket as tscm:
                    # Reset backoff on success
                    retry_delay = 1
                    while True:
                        msg = await tscm.recv()
                        
                        # INTEGRITY CHECK: Validar estructura base del mensaje
                        if not msg or 'data' not in msg or 'k' not in msg['data']:
                            continue
                            
                        monitor.update("WebSocket", "Receiving Data")
                        await self.process_socket_message(msg)
                        
            except asyncio.CancelledError:
                logger.info("WebSocket: Shutdown signal received. Closing gracefully.")
                raise
            except Exception as e:
                # EXPONENTIAL BACKOFF RECONNECTION (Rule 3.3)
                # PROFESSOR METHOD:
                # QUÉ: Algoritmo de reintento con espera geométrica.
                # POR QUÉ: Evita saturar la red (DoS) y el API de Binance tras desconexiones masivas.
                # CÓMO: Multiplicamos el delay por 2 en cada fallo, con un tope de 60s.
                logger.error(f"WebSocket Error: {e}. Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(60, retry_delay * 2) 
            finally:
                # Cleanup guaranteed
                logger.info("WebSocket: Executing cleanup...")
                if hasattr(self, 'bsm') and self.bsm:
                    # Closing bsm might fail if loop is closed? No, it's async.
                    # await self.bsm.close() # BSM doesn't always have close()?
                    pass # python-binance BSM relies on client close
                if hasattr(self, 'client') and self.client:
                    await self.client.close_connection()
                logger.info("WebSocket: Cleanup complete.")
                
        self.last_event_time = {} # Throttling cache

    async def process_socket_message(self, msg):
        """
        Processes incoming WebSocket messages.
        """
        try:
            if 'data' not in msg:
                return
                
            data = msg['data']
            # Parse kline data
            # Event type: e, Event time: E, Symbol: s, Kline: k
            kline = data['k']
            symbol = kline['s'] # e.g. BTCUSDT
            
            # Convert symbol back to internal format if needed
            # Our internal format might be BTC/USDT or BTCUSDT. 
            # Let's try to match it to self.symbol_list
            internal_symbol = symbol
            if symbol not in self.symbol_list:
                # Try adding slash
                if symbol.endswith('USDT'):
                    test_sym = f"{symbol[:-4]}/USDT"
                    if test_sym in self.symbol_list:
                        internal_symbol = test_sym
            
            is_closed = kline['x'] # Boolean: Is this kline closed?
            
            # Extract data
            timestamp = pd.to_datetime(kline['t'], unit='ms')
            open_price = float(kline['o'])
            high_price = float(kline['h'])
            low_price = float(kline['l'])
            close_price = float(kline['c'])
            volume = float(kline['v'])
            
            # DATA QUALITY FILTER (Rule 3.2 - Refined)
            # If the bar is NOT closed and has no movement/volume, skip to save CPU.
            # But if it IS closed, we MUST record it to keep the time-series contiguous.
            if not is_closed:
                if volume <= 0 or (high_price == low_price):
                    return
            elif volume <= 0:
                logger.debug(f"🕳️ [Loader] Liquidity Hole in {internal_symbol}: Zero volume bar recorded for time continuity.")

            bar_data = {
                'symbol': internal_symbol,
                'datetime': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            # GAP DETECTION (Rule 3.2)
            # Detección de saltos temporales en la secuencia de velas
            with self._data_lock:
                if self.latest_data[internal_symbol]:
                    last_ts = self.latest_data[internal_symbol][-1]['datetime']
                    time_diff = (timestamp - last_ts).total_seconds()
                    
                    if time_diff > 65 and is_closed: # Más de 65s para velas de 1m
                        logger.warning(f"🚨 GAP DETECTED in {internal_symbol}: {time_diff}s interval. Data might be missing.")
            
            # Update latest_data (Thread-Safe)
            is_new_bar = False
            with self._data_lock:
                if not self.latest_data[internal_symbol]:
                    self.latest_data[internal_symbol].append(bar_data)
                    is_new_bar = True
                elif self.latest_data[internal_symbol][-1]['datetime'] != timestamp:
                    # New bar
                    self.latest_data[internal_symbol].append(bar_data)
                    # Limit buffer handled by deque(maxlen) automatically!
                    is_new_bar = True
                else:
                    # Update current bar
                    self.latest_data[internal_symbol][-1] = bar_data
            
            # Trigger Market Event (THROTTLED)
            # For scalping, we want updates but not at 100Hz.
            # Only trigger if bar is closed OR at most once every 2 seconds per symbol
            should_trigger = is_closed
            now_ts = time.time()
            
            # --- VOLATILITY BYPASS (High Frequency Fix) ---
            # Calculates instant price change since last update.
            # If move > 0.05% (0.0005), FORCE TRIGGER immediately.
            if not should_trigger:
                try:
                    current_price = close_price
                    last_price = 0.0
                    
                    # Peep at last price safely
                    if internal_symbol in self.latest_data and self.latest_data[internal_symbol]:
                        # No need to lock just for reading a float in python (atomic-ish) 
                        # but let's be safe if we were doing more. 
                        # Ideally we use the lock, but we are inside an async method and _data_lock is a thread lock.
                        # Since we are just reading the last appended value:
                        last_price = self.latest_data[internal_symbol][-1]['close']
                    
                    if last_price > 0:
                        pct_change = abs((current_price - last_price) / last_price)
                        if pct_change >= 0.0005: # 0.05% threshold
                            should_trigger = True
                            # logger.info(f"⚡ VOLATILITY BYPASS: {internal_symbol} moved {pct_change*100:.3f}% in <2s")
                except Exception:
                    pass

            if not should_trigger:
                last_t = self.last_event_time.get(internal_symbol, 0)
                if now_ts - last_t > 2.0:
                    should_trigger = True
                    self.last_event_time[internal_symbol] = now_ts
            
            if should_trigger:
                # OPTIMIZED: Trigger with payload
                self.events_queue.put(MarketEvent(
                    symbol=internal_symbol,
                    close_price=close_price,
                    timestamp=datetime.now(timezone.utc)
                ))
            
            if is_closed:
                logger.info(f"🌊 WebSocket Closed Bar: {internal_symbol} @ {close_price}")
                
        except Exception as e:
            logger.error(f"WebSocket Message Error: {e}")

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
            
        logger.info(f"🔄 DataProvider: Updating Symbol List. Added: {added}, Removed: {removed}")
        
        # 1. Update list
        self.symbol_list = new_symbols
        
        # 2. Cleanup Removed
        with self._data_lock:
            for s in removed:
                if s in self.latest_data: del self.latest_data[s]
                if s in self.latest_data_1h: del self.latest_data_1h[s]
                if s in self.latest_data_5m: del self.latest_data_5m[s]
                if s in self.latest_data_15m: del self.latest_data_15m[s]
        
        # 3. Init New
        for s in added:
            with self._data_lock:
                self.latest_data[s] = deque(maxlen=2000)
                self.latest_data_1h[s] = deque(maxlen=500)
                self.latest_data_5m[s] = deque(maxlen=500)
                self.latest_data_15m[s] = deque(maxlen=500)
            
            # Fetch history for new symbols in parallel
            self.executor.submit(self._fetch_deep_history_worker, s, '1m', 25, self.latest_data)
            self.executor.submit(self._fetch_deep_history_worker, s, '1h', 250, self.latest_data_1h)
            self.executor.submit(self._fetch_deep_history_worker, s, '5m', 100, self.latest_data_5m)
            self.executor.submit(self._fetch_deep_history_worker, s, '15m', 100, self.latest_data_15m)

        # 4. Restart WebSocket
        logger.info("📡 DataProvider: Restarting WebSocket to apply new subscriptions...")
        await self.stop_socket()
        # The background loop in start_socket will automatically reconnect

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
