import ccxt
import pandas as pd
import time
from .data_provider import DataProvider
from core.events import MarketEvent
from config import Config  # Import Config

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
            print("Binance Loader: Configured for Testnet/Demo (Manual URLs)")
        
        # Storage for latest bars
        self.latest_data = {s: [] for s in symbol_list}
        self.latest_data_1h = {s: [] for s in symbol_list}  # 1h candles storage
        self.latest_data_5m = {s: [] for s in symbol_list}  # NEW: 5m candles storage
        self.latest_data_15m = {s: [] for s in symbol_list}  # NEW: 15m candles storage
        
        # Fetch initial history at startup
        self.fetch_initial_history()
        self.fetch_initial_history_1h()
        self.fetch_initial_history_5m()  # NEW
        self.fetch_initial_history_15m()  # NEW

    def fetch_initial_history(self):
        """
        Fetches ~25 hours of historical data (1m candles) from Binance API.
        Optimized for RAM usage while maintaining sufficient data for ML training (~1500 bars).
        """
        print("Fetching historical data from Binance (42 hours)...")
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
                print(f"Loaded {len(self.latest_data[s])} historical bars for {s}")
                
            except Exception as e:
                print(f"Failed to fetch history for {s}: {e}")

    def fetch_initial_history_1h(self):
        """
        Fetches ~200 hours of historical data (1h candles) for trend analysis.
        """
        print("Fetching 1h historical data from Binance (200 hours)...")
        limit = 500
        hours = 200
        
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
                print(f"Loaded {len(self.latest_data_1h[s])} 1h bars for {s}")
                
            except Exception as e:
                print(f"Failed to fetch 1h history for {s}: {e}")

    def get_latest_bars(self, symbol, n=1):
        """
        Returns the last N bars from the latest_symbol list.
        """
        try:
            # ZERO-TRUST: Return a DEEP COPY to prevent strategies from corrupting shared data
            import copy
            bars_list = copy.deepcopy(self.latest_data[symbol])
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        
        return bars_list[-n:]

    def get_latest_bars_1h(self, symbol, n=1):
        """
        Returns the last N 1h bars.
        """
        try:
            bars_list = self.latest_data_1h[symbol]
        except KeyError:
            print("That symbol is not available in the 1h data set.")
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
                if not self.latest_data[s]:
                    self.latest_data[s].append(bar_data)
                elif self.latest_data[s][-1]['datetime'] != timestamp:
                    # New bar detected
                    self.latest_data[s].append(bar_data)
                    
                    # RAM OPTIMIZATION: Limit buffer size to prevent memory growth
                    if len(self.latest_data[s]) > 2000:
                        self.latest_data[s] = self.latest_data[s][-2000:]
                    
                    print(f"New Bar for {s}: {timestamp} - Close: {bar_data['close']}")
                    self.events_queue.put(MarketEvent())
                else:
                    # SAME BAR: Update it in place! (CRITICAL FIX)
                    # Previously we ignored updates, so we only had the "Open" price stored as "Close"
                    self.latest_data[s][-1] = bar_data
                    # We don't necessarily need to fire a MarketEvent on every tick update to avoid spam,
                    # but for real-time ML, we might want to.
                    # For now, let's fire it so strategies see the price moving.
                    # self.events_queue.put(MarketEvent()) # Uncomment if we want tick-level updates
                    pass
                
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
                print(f"Error fetching data for {s}: {e}")

    def fetch_initial_history_5m(self):
        """
        Fetches 5-minute candles for multi-timeframe analysis.
        Loads last 150 bars (~12.5 hours) to support ML features.
        """
        print("Fetching 5m historical data from Binance...")
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
                print(f"Loaded {len(self.latest_data_5m[s])} 5m bars for {s}")
                
                # Rate limit
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Failed to fetch 5m history for {s}: {e}")
    
    def fetch_initial_history_15m(self):
        """
        Fetches 15-minute candles for multi-timeframe analysis.
        Loads last 100 bars (~25 hours) to support ML features.
        """
        print("Fetching 15m historical data from Binance...")
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
                print(f"Loaded {len(self.latest_data_15m[s])} 15m bars for {s}")
                
                # Rate limit
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Failed to fetch 15m history for {s}: {e}")
    
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
