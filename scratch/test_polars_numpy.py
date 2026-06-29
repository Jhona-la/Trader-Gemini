import numpy as np
import polars as pl
import time

dtype = np.dtype([
    ('timestamp', 'i8'), ('open', 'f8'), ('high', 'f8'), 
    ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')
])

arr = np.zeros(250, dtype=dtype)
arr['timestamp'] = np.arange(250)

# Method 1: pl.DataFrame(arr)
start = time.perf_counter_ns()
for _ in range(100):
    df = pl.DataFrame(arr)
print("pl.DataFrame(arr):", (time.perf_counter_ns() - start) / 100, "ns")

# Method 2: Convert to dict of arrays
start = time.perf_counter_ns()
for _ in range(100):
    df2 = pl.DataFrame({name: arr[name] for name in arr.dtype.names})
print("pl.DataFrame(dict):", (time.perf_counter_ns() - start) / 100, "ns")
