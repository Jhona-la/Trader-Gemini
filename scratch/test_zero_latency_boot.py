import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest_infra import fetch_multi_symbol_data

if __name__ == "__main__":
    t0 = time.time()
    data = fetch_multi_symbol_data(["BTC/USDT"], days=3)
    t1 = time.time()
    print(f"Data shape: {data['BTC/USDT'].shape}")
    print(f"Time taken: {(t1 - t0)*1000:.2f} ms")
