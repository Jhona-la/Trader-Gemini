import os
from binance.client import Client
from config import Config
import pandas as pd

def test_fetch_history():
    print("Testing python-binance historical data fetch...")
    
    # Initialize Client
    api_key = Config.BINANCE_API_KEY
    api_secret = Config.BINANCE_SECRET_KEY
    
    if Config.BINANCE_USE_TESTNET:
        api_key = Config.BINANCE_TESTNET_API_KEY
        api_secret = Config.BINANCE_TESTNET_SECRET_KEY
        print("Using Testnet Keys")
    
    # Create Client
    client = Client(api_key, api_secret, testnet=Config.BINANCE_USE_TESTNET)
    
    # Fetch 1m candles for BTC/USDT
    symbol = "BTCUSDT"
    print(f"Fetching 1m klines for {symbol}...")
    
    try:
        # fetch last 5 candles
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=5)
        
        for k in klines:
            # [Open Time, Open, High, Low, Close, Volume, Close Time, Quote Asset Volume, Number of Trades, Taker Buy Base Asset Volume, Taker Buy Quote Asset Volume, Ignore]
            ts = k[0]
            o = k[1]
            h = k[2]
            l = k[3]
            c = k[4]
            v = k[5]
            dt = pd.to_datetime(ts, unit='ms')
            print(f"Time: {dt}, Open: {o}, High: {h}, Low: {l}, Close: {c}, Vol: {v}")
            
        print("✅ Fetch successful")
        
    except Exception as e:
        print(f"❌ Fetch failed: {e}")

if __name__ == "__main__":
    test_fetch_history()
