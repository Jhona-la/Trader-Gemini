import polars as pl
import numpy as np
from .data_provider import DataProvider
from core.events import MarketEvent

class HistoricalCSVData(DataProvider):
    """
    Simulates a real-time feed by reading from CSV files line by line using Polars.
    """
    def __init__(self, events_queue, csv_dir="data/historical", symbol_list=['BTC/USDT', 'ETH/USDT']):
        self.events_queue = events_queue
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        
        self.latest_data = {s: [] for s in symbol_list}
        self.continue_backtest = True
        
        self._load_data()

    def _load_data(self):
        # Load all DFs
        dfs = {}
        for s in self.symbol_list:
            safe_symbol = s.replace('/', '_')
            path = f"{self.csv_dir}/{safe_symbol}_1m.csv"
            try:
                # Use Polars to read csv
                df = pl.read_csv(path, try_parse_dates=True)
                # Sort by datetime to ensure order
                if 'datetime' in df.columns:
                    df = df.sort('datetime')
                dfs[s] = df
            except Exception:
                print(f"Warning: No data for {s}")
                dfs[s] = pl.DataFrame()
        
        # In Polars, simulating a streaming backtest row-by-row is inefficient,
        # but to keep the old interface, we convert to list of dicts or iterators.
        self.data_generators = {}
        for s in dfs:
            if not dfs[s].is_empty():
                self.data_generators[s] = iter(dfs[s].iter_rows(named=True))
            else:
                self.data_generators[s] = iter([])

    def get_latest_bars(self, symbol, n=1):
        try:
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        return bars_list[-n:]

    def update_bars(self):
        try:
            any_data = False
            for s in self.symbol_list:
                try:
                    row = next(self.data_generators[s])
                    
                    bar_data = {
                        'symbol': s,
                        'datetime': row.get('datetime', None),
                        'open': row.get('open', 0.0),
                        'high': row.get('high', 0.0),
                        'low': row.get('low', 0.0),
                        'close': row.get('close', 0.0),
                        'volume': row.get('volume', 0.0)
                    }
                    
                    self.latest_data[s].append(bar_data)
                    any_data = True
                    
                except StopIteration:
                    self.continue_backtest = False
                    return

            if any_data:
                self.events_queue.put(MarketEvent())
            else:
                self.continue_backtest = False
                
        except Exception as e:
            print(f"Backtest Error: {e}")
            self.continue_backtest = False
