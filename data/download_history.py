import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def download_data(symbol, timeframe='1m', days=30):
    print(f"Downloading {days} days of {timeframe} data for {symbol}...")
    
    exchange = ccxt.binance()
    
    # Calculate start time in milliseconds
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    all_candles = []
    
    while since < exchange.milliseconds():
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles:
                break
            
            since = candles[-1][0] + 1 # Next start time
            all_candles += candles
            print(f"Fetched {len(candles)} candles, total: {len(all_candles)}")
            
            # Sleep to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error: {e}")
            break
            
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to CSV
    safe_symbol = symbol.replace('/', '_')
    directory = "data/historical"
    os.makedirs(directory, exist_ok=True)
    filepath = f"{directory}/{safe_symbol}_{timeframe}.csv"
    
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")

if __name__ == "__main__":
    # Download data for our pairs
    download_data('BTC/USDT', days=7) # 1 week of data for quick test
    download_data('ETH/USDT', days=7)
