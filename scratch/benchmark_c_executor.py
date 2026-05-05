import time
import hmac
import hashlib

# Standard python hashing
def py_sign(query: str, secret: bytes):
    return hmac.new(secret, query.encode('utf-8'), hashlib.sha256).hexdigest()

try:
    from execution.c_executor import FastBinanceSigner
    signer = FastBinanceSigner("DUMMY_KEY", "DUMMY_SECRET_KEY_FOR_TESTING_PURPOSES_ONLY")
except ImportError as e:
    print(f"Could not load Cython module: {e}")
    exit(1)

query = "symbol=BTCUSDT&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1.000&price=65000.00&timestamp=1700000000000"
secret_bytes = b"DUMMY_SECRET_KEY_FOR_TESTING_PURPOSES_ONLY"

# Warmup
for _ in range(100):
    py_sign(query, secret_bytes)
    signer.sign_query(query)

N = 100000

# Benchmark Python
start = time.perf_counter_ns()
for _ in range(N):
    py_sign(query, secret_bytes)
end = time.perf_counter_ns()
py_time = (end - start) / N

# Benchmark Cython
start = time.perf_counter_ns()
for _ in range(N):
    signer.sign_query(query)
end = time.perf_counter_ns()
cy_time = (end - start) / N

# Build payload benchmark
start = time.perf_counter_ns()
for _ in range(N):
    signer.build_fapi_order("BTCUSDT", "BUY", "LIMIT", 1.0, 65000.0)
end = time.perf_counter_ns()
cy_build_time = (end - start) / N

print(f"Python hmac latency:      {py_time / 1000:.2f} µs")
print(f"Cython sign_query:        {cy_time / 1000:.2f} µs")
print(f"Cython build_fapi_order:  {cy_build_time / 1000:.2f} µs")
