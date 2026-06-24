import time
import json
import numpy as np
import threading
from core.rust_parser_bridge import ffi_fast_parse_depth

raw_ws_msgs = [
    json.dumps({
        "e": "depthUpdate", "E": 123456789, "s": "BTCUSDT",
        "U": 100, "u": 200,
        "b": [["60000.5", "1.5"], ["60000.0", "0.5"]],
        "a": [["60001.0", "2.0"]]
    }).encode('utf-8') for _ in range(1000)
]

shared_buffer = np.zeros(6, dtype=np.float64)
lock = threading.Lock()

errors = 0

def worker():
    global errors
    for msg in raw_ws_msgs:
        try:
            with lock:
                arr = ffi_fast_parse_depth(msg)
                # arr should have [E, U, best_bid_p, best_bid_v, best_ask_p, best_ask_v]
                if arr[0] != 123456789 or arr[1] != 100 or arr[2] != 60000.5 or arr[4] != 60001.0:
                    errors += 1
                np.copyto(shared_buffer, arr)
        except Exception as e:
            errors += 1

threads = [threading.Thread(target=worker) for _ in range(10)]
t0 = time.perf_counter()

for t in threads: t.start()
for t in threads: t.join()

t1 = time.perf_counter()
print(f"Phase 1 - Ingestion Audit:")
print(f"Processed 10,000 Depth Updates (10 threads)")
print(f"Time: {(t1-t0)*1000:.2f} ms")
print(f"Errors / Collisions: {errors}")
print(f"Shared Buffer Final State: {shared_buffer}")
