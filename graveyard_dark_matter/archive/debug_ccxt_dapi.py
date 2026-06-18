import ccxt
from config import Config

def debug():
    print("--- Debugging CCXT Configuration ---")
    
    options = {'adjustForTimeDifference': True, 'defaultType': 'future'}
    exchange = ccxt.binance({'options': options})
    
    # Apply MANUAL CONFIGURATION exactly as in binance_loader.py
    exchange.urls['api'] = {
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
    
    print(f"Sandbox Mode: {exchange.sandbox}")
    print(f"Default Type: {exchange.options['defaultType']}")
    print(f"URLs keys: {list(exchange.urls['api'].keys())}")
    
    print("\nAttempting to fetch markets...")
    try:
        markets = exchange.load_markets()
        print(f"✅ Successfully loaded {len(markets)} markets.")
    except Exception as e:
        print(f"❌ Failed to load markets: {e}")

    print("\nAttempting to fetch OHLCV for BTC/USDT...")
    try:
        candles = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=5)
        print(f"✅ Successfully fetched {len(candles)} candles.")
    except Exception as e:
        print(f"❌ Failed to fetch OHLCV: {e}")

if __name__ == "__main__":
    debug()
