import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum.mmap_storage import get_quantum_lake
from config import Config

def seed_qbin():
    print("🚀 Seed_qbin: Migrating from Parquet to Quantum Data Lake (.qbin)")
    cache_dir = os.path.join(Config.BASE_DIR, "data/cache_parquet")
    if not os.path.exists(cache_dir):
        print(f"❌ No parquet cache found at {cache_dir}")
        return
        
    lake = get_quantum_lake()
    
    for filename in os.listdir(cache_dir):
        if not filename.endswith("_master.parquet"):
            continue
            
        symbol = filename.split("_master")[0].upper()
        filepath = os.path.join(cache_dir, filename)
        
        try:
            print(f"📥 Loading {symbol} from parquet...")
            df = pd.read_parquet(filepath)
            
            # The parquet files have 'timestamp' as index. Let's reset it if so.
            if 'timestamp' not in df.columns:
                df = df.reset_index()
                
            # Drop timezone info if it exists and convert to ms
            ts_series = df['timestamp']
            if ts_series.dt.tz is not None:
                ts_series = ts_series.dt.tz_convert(None)
            
            # Convert to ms
            timestamps_ms = ts_series.astype('int64') // 1_000_000
            
            ohlcv = np.zeros((len(df), 5), dtype=np.float32)
            ohlcv[:, 0] = df['open'].values
            ohlcv[:, 1] = df['high'].values
            ohlcv[:, 2] = df['low'].values
            ohlcv[:, 3] = df['close'].values
            ohlcv[:, 4] = df['volume'].values
            
            pool = lake._get_pool(symbol)
            pool.inject_bulk(timestamps_ms.values, ohlcv)
            pool.flush()
            print(f"✅ Migrated {len(df)} candles for {symbol} to MMAP")
        except Exception as e:
            print(f"❌ Error migrating {symbol}: {e}")

if __name__ == '__main__':
    seed_qbin()
