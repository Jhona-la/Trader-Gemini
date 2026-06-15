import sys
import os
import queue
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.ml_strategy import UniversalEnsembleStrategy as MLStrategy
from config import Config

class MockData:
    def __init__(self):
        # Generate 2000 rows of dummy data so it has enough bars
        np.random.seed(42)
        dates = pd.date_range("2026-06-01", periods=2000, freq='1min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(2000).cumsum() + 100,
            'high': np.random.randn(2000).cumsum() + 102,
            'low': np.random.randn(2000).cumsum() + 98,
            'close': np.random.randn(2000).cumsum() + 100,
            'volume': np.random.randint(1, 100, 2000),
            'quote_asset_volume': np.random.randint(100, 1000, 2000),
            'number_of_trades': np.random.randint(10, 50, 2000),
            'taker_buy_base_asset_volume': np.random.randint(1, 50, 2000),
            'taker_buy_quote_asset_volume': np.random.randint(10, 500, 2000)
        })
        # Set datetime index
        df.set_index('timestamp', inplace=True)
        self.df = df

    def get_historical_data(self, symbol, timeframe, limit=1000):
        # Must return dict of arrays or dataframe
        return self.df.tail(limit)
        
    def get_latest_bars(self, symbol, timeframe, lookback):
        return self.df.tail(lookback)

try:
    events_queue = queue.Queue()
    data_provider = MockData()
    ml = MLStrategy(
        data_provider=data_provider,
        events_queue=events_queue,
        symbol="BTC/USDT",
        lookback=2000,
        sentiment_loader=None,
        portfolio=None,
        risk_manager=None,
        horizon="SCALPING",
        models_dir=".",
        db_path=":memory:"
    )
    print("SUCCESS INIT ML")
    
    success = ml.retrain(force=True)
    print(f"TRAIN SUCCESS: {success}")
except Exception as e:
    import traceback
    traceback.print_exc()
