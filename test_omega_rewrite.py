import os
import ctypes
import time
from strategies.nano_technical import NanoTechnicalStrategy

# --- 1. Test the Cython Nano Technical Strategy ---
print("=== TESTING NATIVE CYTHON FEATURE STRATEGY ===")
strategy = NanoTechnicalStrategy(strength_threshold=0.3, adx_threshold=25.0)

# Simulate 100 closes
closes = [50000.0 + i * 10 for i in range(100)]
start_time = time.perf_counter_ns()
# Generates signal natively without Python dictionaries/garbage collection
signal = strategy.generate_signal_wrapper(closes, adx=30.0, rsi=25.0)
end_time = time.perf_counter_ns()

print(f"Generated Native Signal: {signal}")
print(f"Latency: {end_time - start_time} nanoseconds")

# --- 2. Test the Rust Networking / Executor FFI ---
print("\n=== TESTING RUST ZERO-LATENCY EXECUTOR ===")
_lib_path = os.path.join(os.path.dirname(__file__), 'core', 'rust_engine', 'target', 'debug', 'quantum_engine.dll')
if os.path.exists(_lib_path):
    lib = ctypes.CDLL(_lib_path)
    
    if hasattr(lib, 'ffi_execute_order'):
        lib.ffi_execute_order.argtypes = [
            ctypes.c_char_p, # api
            ctypes.c_char_p, # secret
            ctypes.c_char_p, # symbol
            ctypes.c_char_p, # side
            ctypes.c_char_p, # type
            ctypes.c_double, # quantity
            ctypes.c_double, # price
        ]
        lib.ffi_execute_order.restype = ctypes.c_bool

        print("Testing FFI Order routing payload (Dry-run mode, invalid API key to prevent real order)")
        api = b"invalid_api"
        secret = b"invalid_secret"
        symbol = b"BTCUSDT"
        side = b"BUY"
        order_type = b"MARKET"
        qty = 0.001
        price = 0.0
        
        start_time_rust = time.perf_counter_ns()
        success = lib.ffi_execute_order(api, secret, symbol, side, order_type, qty, price)
        end_time_rust = time.perf_counter_ns()
        
        print(f"Rust FFI Call Dispatch success: {success}")
        print(f"Python-to-Rust Dispatch Latency: {end_time_rust - start_time_rust} nanoseconds")
else:
    print(f"Rust Engine DLL not found yet at {_lib_path}. Cargo build still required.")
