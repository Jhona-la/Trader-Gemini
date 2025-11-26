import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

class HistoricLoader:
    def __init__(self, symbols, days=365, timeframe='1m'):
        self.symbols = symbols
        self.days = days
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        self.data_dir = "data/backtest_data"
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data(self):
        print(f"Starting download of {self.days} days of {self.timeframe} data...")
        
        limit = 1000
        total_candles = self.days * 24 * 60 # Approx for 1m
        
        for symbol in self.symbols:
            print(f"Downloading {symbol}...")
            filename = os.path.join(self.data_dir, f"{symbol.replace('/', '_')}.csv")
            
            # Check if already exists
            if os.path.exists(filename):
                print(f"Data for {symbol} already exists. Skipping.")
                continue
                
            all_candles = []
            since = self.exchange.milliseconds() - (self.days * 24 * 60 * 60 * 1000)
            
            while True:
                try:
                    candles = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=limit, since=since)
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    since = candles[-1][0] + 60000 # Next minute
                    
                    print(f"Fetched {len(all_candles)} candles...", end='\r')
                    
                    if len(candles) < limit:
                        break
                        
                    # Rate limit
                    time.sleep(0.1)
                    
                    # Safety break for very long downloads
                    if len(all_candles) >= total_candles:
                        break
                        
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(5)
            
            # Save to CSV
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.to_csv(filename, index=False)
            print(f"\nSaved {len(df)} rows to {filename}")

if __name__ == "__main__":
    # Load config to get symbols
    # Hack: Import config from parent directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config
    
    loader = HistoricLoader(Config.TRADING_PAIRS, days=30) # Start with 30 days for speed
    loader.fetch_data()
