import pandas as pd
import numpy as np
from .data_provider import DataProvider
from core.events import MarketEvent

class HistoricalCSVData(DataProvider):
    """
    Simulates a real-time feed by reading from CSV files line by line.
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
                df = pd.read_csv(path)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                dfs[s] = df
            except FileNotFoundError:
                print(f"Warning: No data for {s}")
                dfs[s] = pd.DataFrame()
        
        # Create a common index
        common_idx = pd.Index([])
        for s in dfs:
            if not dfs[s].empty:
                common_idx = common_idx.union(dfs[s].index)
        
        self.timeline = common_idx.sort_values()
        
        # Reindex and create iterators
        self.data_generators = {}
        for s in dfs:
            if not dfs[s].empty:
                aligned = dfs[s].reindex(self.timeline, method='ffill')
                self.data_generators[s] = aligned.itertuples()
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
            # We assume all generators are aligned by _load_data
            # So we just step them all forward
            
            any_data = False
            for s in self.symbol_list:
                try:
                    row = next(self.data_generators[s])
                    
                    # Check for NaN (if ffill failed at start)
                    if pd.isna(row.close):
                        continue
                        
                    bar_data = {
                        'symbol': s,
                        'datetime': row.Index,
                        'open': row.open,
                        'high': row.high,
                        'low': row.low,
                        'close': row.close,
                        'volume': row.volume
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
