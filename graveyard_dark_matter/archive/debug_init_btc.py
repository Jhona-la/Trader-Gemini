
import sys
import os
import logging

# Configure basic logging to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock DataProvider and EventsQueue for minimal init
class MockDataProvider:
    def get_latest_bars(self, symbol, n=100):
        import pandas as pd
        import numpy as np
        # Return dummy dataframe
        df = pd.DataFrame({
            'open': np.random.randn(n) + 50000,
            'high': np.random.randn(n) + 50100,
            'low': np.random.randn(n) + 49900,
            'close': np.random.randn(n) + 50000,
            'volume': np.random.randn(n) + 100,
            'timestamp': pd.date_range(end='now', periods=n, tz='UTC')
        })
        return df

    def register_callback(self, *args, **kwargs):
        pass

class MockQueue:
    def put(self, item):
        pass

try:
    from config import Config
    print(f"✅ Config Imported.")
    print(f"📋 TRADING_PAIRS type: {type(Config.TRADING_PAIRS)}")
    print(f"📋 TRADING_PAIRS len: {len(Config.TRADING_PAIRS)}")
    print(f"📋 First 5 pairs: {Config.TRADING_PAIRS[:5]}")
    
    if "BTC/USDT" in Config.TRADING_PAIRS:
        print(f"✅ BTC/USDT is in TRADING_PAIRS")
    else:
        print(f"❌ BTC/USDT is NOT in TRADING_PAIRS")

    from strategies.ml_strategy import UniversalEnsembleStrategy
    print(f"✅ MLStrategy Imported.")

    # Try init
    data_provider = MockDataProvider()
    events_queue = MockQueue()
    
    print(f"🔄 Attempting init for BTC/USDT...")
    strategy = UniversalEnsembleStrategy(
        data_provider=data_provider,
        events_queue=events_queue,
        symbol="BTC/USDT",
        lookback=50
    )
    print(f"✅ Init Successful: {strategy.symbol}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
