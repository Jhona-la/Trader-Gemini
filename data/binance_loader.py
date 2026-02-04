# import ccxt  <-- REMOVED (Rule 3.1 Separation of Concerns)
from binance.client import Client # Synchronous Client for REST calls
from binance.enums import *
import pandas as pd
import time
from .data_provider import DataProvider
from core.events import MarketEvent
from config import Config  # Import Config
from utils.logger import logger
from utils.debug_tracer import trace_execution
from utils.thread_monitor import monitor
import asyncio
from binance import AsyncClient, BinanceSocketManager
from concurrent.futures import ThreadPoolExecutor, as_completed

class BinanceData(DataProvider):
    def __init__(self, events_queue, symbol_list):
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        
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
        except Exception as e:
            logger.error(f"❌ Binance REST API Connection Failed: {e}")
        
        # Storage for latest bars
        self.latest_data = {s: [] for s in self.symbol_list}
        self.latest_data_1h = {s: [] for s in self.symbol_list}  # 1h candles storage
        self.latest_data_5m = {s: [] for s in self.symbol_list}  # NEW: 5m candles storage
        self.latest_data_15m = {s: [] for s in self.symbol_list}  # NEW: 15m candles storage
        
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

    def get_latest_bars(self, symbol, n=1):
        """
        Returns the last N bars from the latest_symbol list.
        Thread-safe deep copy.
        """
        try:
            with self._data_lock:
                # ZERO-TRUST: Return a DEEP COPY to prevent strategies from corrupting shared data
                import copy
                bars_list = copy.deepcopy(self.latest_data[symbol])
        except KeyError:
            logger.warning("That symbol is not available in the historical data set.")
            raise
        
        return bars_list[-n:]

    def fetch_initial_history(self):
        """
        Fetches ~25 hours of historical data (1m candles) from Binance API.
        Optimized for RAM usage while maintaining sufficient data for ML training (~1500 bars).
        """
        time_needed = Config.Strategies.ML_LOOKBACK_BARS * 1.2 # 20% buffer
        hours = int(time_needed / 60) + 5 # Convert minutes to hours + margin
        logger.info(f"Fetching historical data from Binance ({hours} hours / ~{time_needed} bars)...")
        limit = 1000
        # hours variable used above
        total_candles = hours * 60
        
        for s in self.symbol_list:
            try:
                all_candles = []
                # Calculate start time
                since = int(time.time() * 1000) - (hours * 60 * 60 * 1000)
                
                while len(all_candles) < total_candles:
                    # candles = self.exchange.fetch_ohlcv(s, timeframe='1m', limit=limit, since=since)
                    sym_clean = s.replace('/', '') # Phase 6 standardized slash removal
                    candles = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit, startTime=since)
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    # Update since to the timestamp of the last candle + 1 minute
                    since = candles[-1][0] + 60000
                    
                    # Respect rate limits
                    time.sleep(0.2)
                    
                    if len(candles) < limit:
                        break
                
                # Process and store
                processed_bars = []
                for c in all_candles:
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
                
                # RAM OPTIMIZATION: Keep meaningful amount but enough for ML
                keep_size = max(Config.Strategies.ML_LOOKBACK_BARS + 500, 2000)
                self.latest_data[s] = processed_bars[-keep_size:]
                logger.info(f"Loaded {len(self.latest_data[s])} historical bars for {s}")
                
            except Exception as e:
                logger.error(f"Failed to fetch history for {s}: {e}")

    def fetch_initial_history_1h(self):
        """
        Fetches ~200 hours of historical data (1h candles) for trend analysis.
        """
        logger.info("Fetching 1h historical data from Binance (250 hours)...")
        limit = 500
        hours = 250 # FIXED: Increased from 200 to 250 to ensure >200 closed candles for EMA-200
        
        for s in self.symbol_list:
            try:
                # Calculate start time: now - 200 hours
                # since = self.exchange.milliseconds() - (hours * 60 * 60 * 1000)
                since = int(time.time() * 1000) - (hours * 60 * 60 * 1000)
                # candles = self.exchange.fetch_ohlcv(s, timeframe='1h', limit=limit, since=since)
                sym_clean = s.replace('/', '')
                candles = self.client_sync.get_klines(symbol=sym_clean, interval=Client.KLINE_INTERVAL_1HOUR, limit=limit, startTime=since)
                
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
                
                self.latest_data_1h[s] = processed_bars
                logger.info(f"Loaded {len(self.latest_data_1h[s])} 1h bars for {s}")
                
            except Exception as e:
                logger.error(f"Failed to fetch 1h history for {s}: {e}")

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
                    with self._data_lock:
                        if not self.latest_data[s]:
                            self.latest_data[s].append(bar_data)
                            is_new_bar = True
                        elif self.latest_data[s][-1]['datetime'] != timestamp:
                            self.latest_data[s].append(bar_data)
                            if len(self.latest_data[s]) > 2000:
                                self.latest_data[s] = self.latest_data[s][-2000:]
                            is_new_bar = True
                        else:
                            self.latest_data[s][-1] = bar_data
                            is_new_bar = False
                            
                    if is_new_bar:
                        logger.info(f"New Bar for {s}: {timestamp} - Close: {bar_data['close']}")
                        self.events_queue.put(MarketEvent())

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
                             if len(self.latest_data_5m[s]) > 200: self.latest_data_5m[s] = self.latest_data_5m[s][-200:]
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
        while True:
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
            
            # DATA QUALITY FILTER (Rule 3.2)
            # PROFESSOR METHOD:
            # QUÉ: Filtro de integridad y velas "Zombie".
            # POR QUÉ: Velas con volumen cero o precio estático (high == low) ensucian los indicadores técnicos.
            # CÓMO: Validamos que volumen > 0 y que el precio no sea exactamente igual en extremos.
            if volume <= 0:
                # Ignorar velas sin actividad para evitar distorsión de RSI/ADX
                return
            
            if high_price == low_price and not is_closed:
                # Mercado congelado, ignorar ticks hasta que haya movimiento
                return

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
                    # Limit buffer
                    keep_size = max(Config.Strategies.ML_LOOKBACK_BARS + 500, 2000)
                    if len(self.latest_data[internal_symbol]) > keep_size:
                        self.latest_data[internal_symbol] = self.latest_data[internal_symbol][-keep_size:]
                    is_new_bar = True
                else:
                    # Update current bar
                    self.latest_data[internal_symbol][-1] = bar_data
            
            # Trigger Market Event (THROTTLED)
            # For scalping, we want updates but not at 100Hz.
            # Only trigger if bar is closed OR at most once every 2 seconds per symbol
            should_trigger = is_closed
            now_ts = time.time()
            if not should_trigger:
                last_t = self.last_event_time.get(internal_symbol, 0)
                if now_ts - last_t > 2.0:
                    should_trigger = True
                    self.last_event_time[internal_symbol] = now_ts
            
            if should_trigger:
                self.events_queue.put(MarketEvent())
            
            if is_closed:
                logger.info(f"🌊 WebSocket Closed Bar: {internal_symbol} @ {close_price}")
                
        except Exception as e:
            logger.error(f"WebSocket Message Error: {e}")

    async def shutdown(self):
        """
        Graceful shutdown for all data resources.
        """
        logger.info("BinanceData: Initiating shutdown...")
        
        # 1. Stop Socket
        await self.stop_socket()
        
        # 2. Cleanup ThreadPool
        logger.info("BinanceData: Closing ThreadPoolExecutor...")
        self.executor.shutdown(wait=True)
        
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
