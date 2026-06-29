import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from queue import Queue
from config import Config
Config.LEAN_MODE = True
Config.CORE_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
from data.binance_loader import BinanceData
from strategies.omni_strategy import OmniStrategy

async def main():
    q = Queue()
    print("Initializing Data Provider...")
    data_provider = BinanceData(q, Config.CORE_SYMBOLS)
    print("Data Provider initialized.")
    print("Initializing OmniStrategy Scalping...")
    omni_scalp = OmniStrategy(data_provider, q, horizon="SCALPING")
    print("OmniStrategy Scalping initialized.")
    print("Initializing OmniStrategy Swing...")
    omni_swing = OmniStrategy(data_provider, q, horizon="SWING")
    print("OmniStrategy Swing initialized.")
    
    # Simulate WSS start
    print("Starting WebSockets...")
    asyncio.create_task(data_provider.start_websockets())
    print("Waiting 10 seconds...")
    await asyncio.sleep(10)
    print("Done waiting.")
    await data_provider.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
