
from unittest.mock import MagicMock
from data.binance_loader import BinanceData

# Mock
print("Monkey patching...")
BinanceData._init_sync_client = lambda self: print("   Mocked _init_sync_client called")
BinanceData.fetch_initial_history = lambda self: print("   Mocked fetch_initial_history called")
BinanceData.fetch_initial_history_1h = lambda self: print("   Mocked fetch_initial_history_1h called")
BinanceData.fetch_initial_history_5m = lambda self: print("   Mocked fetch_initial_history_5m called")
BinanceData.fetch_initial_history_15m = lambda self: print("   Mocked fetch_initial_history_15m called")

print("Instantiating...")
try:
    loader = BinanceData(None, ['BTC/USDT'])
    print("Instance created.")
    print(f"Has buffers_1m? {hasattr(loader, 'buffers_1m')}")
except Exception as e:
    print(f"CRASHED: {e}")
    import traceback
    traceback.print_exc()
