import pandas as pd
import os

path = 'data/cache_parquet'
pairs = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','DOTUSDT','AVAXUSDT','LINKUSDT']

for p in pairs:
    fpath = os.path.join(path, f"{p}_1m.parquet")
    if os.path.exists(fpath):
        df = pd.read_parquet(fpath)
        print(f"{p}: {len(df)} bars = {len(df)/1440:.1f}D")
    else:
        print(f"{p}: NOT FOUND")
