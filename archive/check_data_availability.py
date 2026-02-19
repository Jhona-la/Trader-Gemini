
import asyncio
import queue
from config import Config
from data.binance_loader import BinanceData
import pandas as pd

async def test_data():
    events = queue.Queue()
    symbols = Config.TRADING_PAIRS
    print(f"Testing {len(symbols)} symbols...")
    
    loader = BinanceData(events, symbols)
    # Give it a moment to start and maybe fetch something (though sync fetch happens in init)
    
    results = []
    for sym in symbols:
        df = loader.get_latest_df(sym, interval='1m', lookback=50)
        found = "YES" if df is not None and len(df) >= 50 else f"NO ({len(df) if df is not None else 'None'})"
        results.append(f"{sym}: {found}")
    
    for r in results:
        print(r)
    
    loader.shutdown()

if __name__ == "__main__":
    asyncio.run(test_data())
