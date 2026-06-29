import ctypes
import os
import numpy as np

_lib_path = os.path.join(os.path.dirname(__file__), 'rust_engine', 'target', 'release', 'quantum_engine.dll')
if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Rust Quantum Engine DLL not found at {_lib_path}. Please compile it.")

lib = ctypes.CDLL(_lib_path)

lib.ffi_parse_binance_depth.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
lib.ffi_parse_binance_depth.restype = ctypes.c_bool

lib.ffi_parse_binance_trade.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
lib.ffi_parse_binance_trade.restype = ctypes.c_bool

# Pre-allocate zero-copy buffers for Thread-Local speed
# Using thread-local storage or pre-allocated globals is fastest for FFI
_depth_buffer = np.zeros(6, dtype=np.float64)
_trade_buffer = np.zeros(5, dtype=np.float64)
_depth_ptr = _depth_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
_trade_ptr = _trade_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def ffi_fast_parse_depth(json_bytes: bytes) -> np.ndarray:
    """
    Parses a Binance @depthUpdate message in Rust in O(1) time.
    Returns a NumPy array [Event_time, lastUpdateId, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty]
    Or None if parsing failed.
    """
    success = lib.ffi_parse_binance_depth(json_bytes, _depth_ptr)
    if success:
        return _depth_buffer
    return None

def ffi_fast_parse_trade(json_bytes: bytes) -> np.ndarray:
    """
    Parses a Binance @trade or @aggTrade message in Rust in O(1) time.
    Returns a NumPy array [Event_time, trade_time, price, qty, is_buyer_maker]
    Or None if parsing failed.
    """
    success = lib.ffi_parse_binance_trade(json_bytes, _trade_ptr)
    if success:
        return _trade_buffer
    return None
