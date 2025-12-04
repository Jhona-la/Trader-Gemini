import ccxt
import pandas as pd
import time
from .data_provider import DataProvider
from core.events import MarketEvent
from config import Config  # Import Config
from utils.logger import logger
import asyncio
from binance import AsyncClient, BinanceSocketManager

class BinanceData(DataProvider):
    def __init__(self, events_queue, symbol_list):
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        
        # Initialize CCXT Binance client with Config options
        options = {
            'adjustForTimeDifference': True,
            'defaultType': 'future' if Config.BINANCE_USE_FUTURES else 'spot'
        }
            
        self.exchange = ccxt.binance({
            'options': options,
            'enableRateLimit': True,  # CRITICAL: Prevent IP bans
            'timeout': 10000          # CRITICAL: 10s timeout to prevent freezing
        })
        
        # Enable Demo/Testnet if configured
        if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
            # MANUAL CONFIGURATION for Futures Testnet/Demo
            # Matches logic in binance_executor.py for consistency
            custom_urls = {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiData': 'https://testnet.binancefuture.com/fapi/v1',
                'dapiPublic': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiPrivate': 'https://testnet.binancefuture.com/dapi/v1',
                'dapiData': 'https://testnet.binancefuture.com/dapi/v1',
                'sapi': 'https://testnet.binance.vision/api/v3',
            }
            self.exchange.urls['api'] = custom_urls
            self.exchange.urls['test'] = custom_urls
            logger.info("Binance Loader: Configured for Testnet/Demo (Manual URLs)")
        
        # Storage for latest bars
        self.latest_data = {s: [] for s in symbol_list}
        self.latest_data_1h = {s: [] for s in symbol_list}  # 1h candles storage
        self.latest_data_5m = {s: [] for s in symbol_list}  # NEW: 5m candles storage
        self.latest_data_15m = {s: [] for s in symbol_list}  # NEW: 15m candles storage
        
        # Thread Safety Lock
        import threading
        self._data_lock = threading.Lock()
        
        # Fetch initial history at startup
        self.fetch_initial_history()
        self.fetch_initial_history_1h()
        self.fetch_initial_history_5m()  # NEW
        self.fetch_initial_history_5m()  # NEW
        self.fetch_initial_history_15m()  # NEW
        
        # Async Client & Socket Manager placeholders
        self.client = None
        self.bsm = None
        self.socket = None
        self.socket_task = None

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
        logger.info("Fetching historical data from Binance (42 hours)...")
        limit = 1000
        hours = 42  # ENHANCED: Increased from 25 to 42 hours (~2520 bars) to support ML training with 2500 bars
        total_candles = hours * 60
        
        for s in self.symbol_list:
            try:
                all_candles = []
                # Calculate start time: now - 25 hours
                since = self.exchange.milliseconds() - (hours * 60 * 60 * 1000)
                
                while len(all_candles) < total_candles:
                    candles = self.exchange.fetch_ohlcv(s, timeframe='1m', limit=limit, since=since)
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
                        'open': c[1],
                        'high': c[2],
                        'low': c[3],
                        'close': c[4],
                        'volume': c[5]
                    })
                
                # RAM OPTIMIZATION: Keep only last 2000 bars maximum
                self.latest_data[s] = processed_bars[-2000:]
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
                since = self.exchange.milliseconds() - (hours * 60 * 60 * 1000)
                candles = self.exchange.fetch_ohlcv(s, timeframe='1h', limit=limit, since=since)
                
                processed_bars = []
                for c in candles:
                    timestamp = pd.to_datetime(c[0], unit='ms')
                    processed_bars.append({
                        'symbol': s,
                        'datetime': timestamp,
                        'open': c[1],
                        'high': c[2],
                        'low': c[3],
                        'close': c[4],
                        'volume': c[5]
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
            bars_list = copy.deepcopy(self.latest_data[symbol])
        except KeyError:
            logger.warning("That symbol is not available in the historical data set.")
            raise
        
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

    def update_bars(self):
        """
        Fetches new bars from Binance and places them into the events queue.
        """
        for s in self.symbol_list:
            try:
                # Fetch OHLCV data (1 minute candles)
                bars = self.exchange.fetch_ohlcv(s, timeframe='1m', limit=2) 
                
                # We take the LAST one (incomplete) for real-time analysis
                latest_bar = bars[-1] 
                
                # Format: [timestamp, open, high, low, close, volume]
                timestamp = pd.to_datetime(latest_bar[0], unit='ms')
                bar_data = {
                    'symbol': s,
                    'datetime': timestamp,
                    'open': latest_bar[1],
                    'high': latest_bar[2],
                    'low': latest_bar[3],
                    'close': latest_bar[4],
                    'volume': latest_bar[5]
                }
                
                # Avoid duplicates (simple check)
                # CRITICAL FIX: Entire if-else MUST be inside lock to prevent race conditions
                is_new_bar = False
                with self._data_lock:
                    if not self.latest_data[s]:
                        self.latest_data[s].append(bar_data)
                        is_new_bar = True
                    elif self.latest_data[s][-1]['datetime'] != timestamp:
                        # New bar detected
                        self.latest_data[s].append(bar_data)
                        
                        # RAM OPTIMIZATION: Limit buffer size to prevent memory growth
                        if len(self.latest_data[s]) > 2000:
                            self.latest_data[s] = self.latest_data[s][-2000:]
                        
                        is_new_bar = True
                    else:
                        # SAME BAR: Update it in place! (NOW INSIDE LOCK)
                        # Previously this was OUTSIDE the lock, causing race conditions
                        self.latest_data[s][-1] = bar_data
                        is_new_bar = False
                
                # Print and event OUTSIDE lock (don't hold lock during I/O)
                if is_new_bar:
                    logger.info(f"New Bar for {s}: {timestamp} - Close: {bar_data['close']}")
                    self.events_queue.put(MarketEvent())
                
                # MULTI-TIMEFRAME: Also update 1h candles
                # Fetch 1h candle (limit=2 to get latest closed or current open)
                bars_1h = self.exchange.fetch_ohlcv(s, timeframe='1h', limit=2)
                if bars_1h:
                    latest_1h = bars_1h[-1]
                    ts_1h = pd.to_datetime(latest_1h[0], unit='ms')
                    
                    # Update if new or same (to update current candle)
                    # For trend analysis, we usually want closed candles, but updating current is fine
                    bar_data_1h = {
                        'symbol': s,
                        'datetime': ts_1h,
                        'open': latest_1h[1],
                        'high': latest_1h[2],
                        'low': latest_1h[3],
                        'close': latest_1h[4],
                        'volume': latest_1h[5]
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
                bars_5m = self.exchange.fetch_ohlcv(s, timeframe='5m', limit=2)
                if bars_5m:
                    latest_5m = bars_5m[-1]
                    ts_5m = pd.to_datetime(latest_5m[0], unit='ms')
                    
                    bar_data_5m = {
                        'symbol': s,
                        'datetime': ts_5m,
                        'open': latest_5m[1],
                        'high': latest_5m[2],
                        'low': latest_5m[3],
                        'close': latest_5m[4],
                        'volume': latest_5m[5]
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
                bars_15m = self.exchange.fetch_ohlcv(s, timeframe='15m', limit=2)
                if bars_15m:
                    latest_15m = bars_15m[-1]
                    ts_15m = pd.to_datetime(latest_15m[0], unit='ms')
                    
                    bar_data_15m = {
                        'symbol': s,
                        'datetime': ts_15m,
                        'open': latest_15m[1],
                        'high': latest_15m[2],
                        'low': latest_15m[3],
                        'close': latest_15m[4],
                        'volume': latest_15m[5]
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
                candles = self.exchange.fetch_ohlcv(s, timeframe='5m', limit=limit)
                
                processed_bars = []
                for c in candles:
                    timestamp = pd.to_datetime(c[0], unit='ms')
                    processed_bars.append({
                        'symbol': s,
                        'datetime': timestamp,
                        'open': c[1],
                        'high': c[2],
                        'low': c[3],
                        'close': c[4],
                        'volume': c[5]
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
                candles = self.exchange.fetch_ohlcv(s, timeframe='15m', limit=limit)
                
                processed_bars = []
                for c in candles:
                    timestamp = pd.to_datetime(c[0], unit='ms')
                    processed_bars.append({
                        'symbol': s,
                        'datetime': timestamp,
                        'open': c[1],
                        'high': c[2],
                        'low': c[3],
                        'close': c[4],
                        'volume': c[5]
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
             
        self.client = await AsyncClient.create(api_key, api_secret, testnet=Config.BINANCE_USE_TESTNET)
        self.bsm = BinanceSocketManager(self.client)
        
        # Create streams for all symbols
        # kline_1m is the stream name for 1-minute candles
        streams = [f"{s.lower().replace('/', '')}@kline_1m" for s in self.symbol_list]
        
        logger.info(f"Subscribing to streams: {streams}")
        self.socket = self.bsm.multiplex_socket(streams)
        
        async with self.socket as tscm:
            while True:
                msg = await tscm.recv()
                await self.process_socket_message(msg)

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
            
            bar_data = {
                'symbol': internal_symbol,
                'datetime': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
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
                    if len(self.latest_data[internal_symbol]) > 2000:
                        self.latest_data[internal_symbol] = self.latest_data[internal_symbol][-2000:]
                    is_new_bar = True
                else:
                    # Update current bar
                    self.latest_data[internal_symbol][-1] = bar_data
            
            # Trigger Market Event on every tick (or just on close?)
            # For HFT/Scalping, we want every tick.
            # But to avoid spamming the event loop, maybe only on significant changes?
            # For now, let's trigger on every update but log only on new bar or close.
            
            self.events_queue.put(MarketEvent())
            
            if is_closed:
                logger.info(f"🌊 WebSocket Closed Bar: {internal_symbol} @ {close_price}")
                
        except Exception as e:
            logger.error(f"WebSocket Message Error: {e}")

    async def stop_socket(self):
        """
        Stops the WebSocket and closes the client.
        """
        if self.client:
            await self.client.close_connection()
            logger.info("Binance WebSocket Client Closed.")
