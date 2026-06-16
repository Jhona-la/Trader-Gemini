import os
import sys

from core.backtest_infra import fetch_binance_data
from core.simulation import SimDataProvider
from joblib import Parallel, delayed
import pickle

if __name__ == '__main__':
    df = fetch_binance_data("BTC/USDT", days=1)
    df.index.name = 'timestamp'
    provider = SimDataProvider({"BTC/USDT": df})
    print("Original names:", provider.arrays["BTC/USDT"].dtype.names)
    
    # Simulate pickle/unpickle
    serialized = pickle.dumps(provider)
    unpickled = pickle.loads(serialized)
    print("Unpickled names:", unpickled.arrays["BTC/USDT"].dtype.names)
