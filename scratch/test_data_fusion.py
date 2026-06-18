import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.backtest_infra import BacktestDataProvider
import polars as pl
from queue import Queue

def test_data_fusion():
    print("Testing data fusion...")
    events_queue = Queue()
    symbol_list = ["BNBUSDT"]
    
    # Load dummy OHLCV data just to initialize
    df = pl.DataFrame({
        "timestamp": [1718668800000],  # Example recent timestamp
        "open": [600.0],
        "high": [605.0],
        "low": [595.0],
        "close": [600.0],
        "volume": [1000.0]
    }).to_pandas()
    import pandas as pd
    df.set_index(pd.DatetimeIndex(pd.to_datetime([1718668800000], unit="ms")), inplace=True)
    
    historical_data = {"BNBUSDT": df}
    
    dp = BacktestDataProvider(events_queue, symbol_list, historical_data)
    dp.current_time_ms = 1718668800000
    
    metrics = dp.get_derivatives_metrics("BNBUSDT")
    print(f"Metrics: {metrics}")
    
    assert metrics["open_interest"] != 0.0, "Open interest is zero!"
    print("✅ Data fusion works.")

if __name__ == "__main__":
    test_data_fusion()
