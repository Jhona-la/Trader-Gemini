import sys
import os
import queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.ml_strategy import MLStrategy
from config import Config

class MockData:
    def get_latest_bars(self, *args, **kwargs):
        return None

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
    print("SUCCESS ML")
except Exception as e:
    import traceback
    traceback.print_exc()
