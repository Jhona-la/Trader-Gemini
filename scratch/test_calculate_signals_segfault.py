import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from queue import Queue
from config import Config
from data.binance_loader import BinanceData
from strategies.omni_strategy import OmniStrategy
from core.events import MarketEvent

Config.LEAN_MODE = True
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
Config.CORE_SYMBOLS = symbols
events_queue = Queue()

print("Initializing Data...")
data_provider = BinanceData(events_queue, symbols)

print("Initializing OmniStrategy SCALPING...")
omni_scalp = OmniStrategy(data_provider, events_queue, horizon="SCALPING")

print("Sending empty MarketEvent...")
event = MarketEvent(symbol="BTC/USDT")
try:
    omni_scalp.calculate_signals(event)
    print("Calculate signals finished successfully (did not segfault).")
except Exception as e:
    print(f"Caught exception: {e}")

print("Done.")
