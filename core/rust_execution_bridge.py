import ctypes
import os
import time

_lib_path = os.path.join(os.path.dirname(__file__), 'rust_engine', 'target', 'release', 'quantum_engine.dll')
if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Rust Quantum Engine DLL not found at {_lib_path}. Please compile it.")

lib = ctypes.CDLL(_lib_path)

if hasattr(lib, 'ffi_sign_binance_payload'):
    lib.ffi_sign_binance_payload.argtypes = [
        ctypes.c_char_p, # secret
        ctypes.c_char_p, # payload
        ctypes.c_char_p, # out buffer
        ctypes.c_size_t  # max len
    ]
    lib.ffi_sign_binance_payload.restype = ctypes.c_bool

if hasattr(lib, 'ffi_start_ws_client'):
    lib.ffi_start_ws_client.argtypes = [ctypes.c_char_p]
    lib.ffi_start_ws_client.restype = ctypes.c_bool

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

if hasattr(lib, 'ffi_poll_ws_event'):
    lib.ffi_poll_ws_event.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.ffi_poll_ws_event.restype = ctypes.c_bool

# Phase 9: Portfolio & Risk FFI
if hasattr(lib, 'ffi_update_portfolio'):
    lib.ffi_update_portfolio.argtypes = [ctypes.c_double]
    
if hasattr(lib, 'ffi_set_position'):
    lib.ffi_set_position.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_double]
    
if hasattr(lib, 'ffi_clear_position'):
    lib.ffi_clear_position.argtypes = [ctypes.c_int32]
    
if hasattr(lib, 'ffi_can_open_position'):
    lib.ffi_can_open_position.argtypes = [ctypes.c_int32, ctypes.c_double, ctypes.c_double]
    lib.ffi_can_open_position.restype = ctypes.c_bool
    
if hasattr(lib, 'ffi_check_drawdown'):
    lib.ffi_check_drawdown.argtypes = [ctypes.c_double]
    lib.ffi_check_drawdown.restype = ctypes.c_bool

# Phase 10: ML Inference FFI
if hasattr(lib, 'ffi_load_nano_forest'):
    lib.ffi_load_nano_forest.argtypes = [ctypes.c_char_p]
    lib.ffi_load_nano_forest.restype = ctypes.c_bool

if hasattr(lib, 'ffi_predict_nano_forest'):
    lib.ffi_predict_nano_forest.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.ffi_predict_nano_forest.restype = ctypes.c_float

_out_buffer = ctypes.create_string_buffer(65) # SHA256 hex is 64 chars + 1 null
_ws_buffer = ctypes.create_string_buffer(1024 * 64) # 64KB for json events

def ffi_sign_order(secret: str, payload: str) -> str:
    """
    Computes HMAC-SHA256 of the payload using the secret in Rust O(1).
    Returns the hex signature string.
    """
    secret_bytes = secret.encode('utf-8')
    payload_bytes = payload.encode('utf-8')
    
    if lib:
        success = lib.ffi_sign_binance_payload(secret_bytes, payload_bytes, _out_buffer, 65)
        if success:
            return _out_buffer.value.decode('utf-8')
    raise RuntimeError("Rust FFI Signature generation failed")

def ffi_execute_order_bridge(api_key: str, secret_key: str, symbol: str, side: str, order_type: str, qty: float, price: float = 0.0) -> bool:
    """
    Triggers native network execution inside Rust without `aiohttp`.
    """
    if lib:
        return lib.ffi_execute_order(
            api_key.encode('utf-8'),
            secret_key.encode('utf-8'),
            symbol.encode('utf-8'),
            side.encode('utf-8'),
            order_type.encode('utf-8'),
            qty,
            price
        )
    return False

def ffi_start_ws_client_bridge(symbols: list) -> bool:
    """
    Boots up the native Rust WebSocket thread.
    """
    symbols_str = ",".join(symbols).encode('utf-8')
    if lib:
        return lib.ffi_start_ws_client(symbols_str)
    return False

def ffi_poll_ws_event_bridge() -> str:
    """
    Polls the Rust networking thread for a new JSON event string.
    Returns empty string if no new event is queued.
    """
    if lib:
        if lib.ffi_poll_ws_event(_ws_buffer, 1024 * 64):
            return _ws_buffer.value.decode('utf-8')
    return ""

# Phase 9: Portfolio & Risk Bridge Wrappers
def ffi_update_portfolio_bridge(usdt_balance: float):
    if lib:
        lib.ffi_update_portfolio(usdt_balance)

def ffi_set_position_bridge(horizon: int, side: int, entry_price: float, qty: float):
    if lib:
        lib.ffi_set_position(horizon, side, entry_price, qty)

def ffi_clear_position_bridge(horizon: int):
    if lib:
        lib.ffi_clear_position(horizon)

def ffi_can_open_position_bridge(horizon: int, requested_qty: float, current_price: float) -> bool:
    if lib:
        return lib.ffi_can_open_position(horizon, requested_qty, current_price)
    return False

def ffi_check_drawdown_bridge(current_price: float) -> bool:
    if lib and hasattr(lib, 'ffi_check_drawdown'):
        return lib.ffi_check_drawdown(current_price)
    return False

# Phase 10: ML Inference Bridge Wrappers
def ffi_load_nano_forest_bridge(path: str) -> bool:
    if lib and hasattr(lib, 'ffi_load_nano_forest'):
        return lib.ffi_load_nano_forest(path.encode('utf-8'))
    return False

def ffi_predict_nano_forest_bridge(features: list) -> float:
    if lib and hasattr(lib, 'ffi_predict_nano_forest'):
        import ctypes
        float_array_type = ctypes.c_float * len(features)
        float_array = float_array_type(*features)
        return lib.ffi_predict_nano_forest(float_array, len(features))
    return 0.5


class RustBinanceSigner:
    """
    Ultra-fast Rust-optimized signature generator for Binance API.
    Bypasses the Python HMAC/Hashlib abstraction layer to build payloads in nanoseconds.
    Drop-in replacement for FastBinanceSigner.
    """
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key

    def sign_query(self, query_string: str) -> str:
        """
        Generates HMAC SHA256 signature for Binance API using Rust Engine.
        """
        return ffi_sign_order(self.secret_key, query_string)
        
    def build_fapi_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float=0.0, timeInForce: str="GTC", reduceOnly: bool=False, positionSide: str="BOTH"):
        """
        Constructs the exact HTTP query and headers for Binance Futures (fapi) order endpoint.
        Returns: (endpoint, query_string, headers)
        """
        timestamp = int(time.time() * 1000)
        params = [
            f"symbol={symbol}",
            f"side={side}",
            f"positionSide={positionSide}",
            f"type={order_type}",
            f"quantity={quantity:.6f}".rstrip('0').rstrip('.'),
            f"newOrderRespType=RESULT",
            f"recvWindow=60000",
            f"timestamp={timestamp}"
        ]
        
        if order_type in ("LIMIT", "STOP", "TAKE_PROFIT"):
            params.append(f"price={price:.6f}".rstrip('0').rstrip('.'))
            params.append(f"timeInForce={timeInForce}")
            
        if reduceOnly:
            params.append("reduceOnly=true")
            
        query_string = "&".join(params)
        
        # ⚡ FFI RUST INVOCATION (Microsecond latency drop)
        signature = self.sign_query(query_string)
        
        final_query = f"{query_string}&signature={signature}"
        
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        return ("/fapi/v1/order", final_query, headers)
