# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import hmac
import hashlib
import time
try:
    import orjson
except ImportError:
    import json as orjson

cdef class FastBinanceSigner:
    """
    Ultra-fast Cython-optimized signature generator for Binance API.
    Bypasses the CCXT abstraction layer to build payloads in microseconds.
    """
    cdef str api_key
    cdef bytes secret_key
    
    def __init__(self, str api_key, str secret_key):
        self.api_key = api_key
        self.secret_key = secret_key.encode('utf-8')

    cpdef str sign_query(self, str query_string):
        """
        Generates HMAC SHA256 signature for Binance API using Python's C-API via Cython.
        """
        cdef bytes query_bytes = query_string.encode('utf-8')
        cdef str signature = hmac.new(self.secret_key, query_bytes, hashlib.sha256).hexdigest()
        return signature
        
    cpdef tuple build_fapi_order(self, str symbol, str side, str order_type, double quantity, double price=0.0, str timeInForce="GTC", bint reduceOnly=False, str positionSide="BOTH"):
        """
        Constructs the exact HTTP query and headers for Binance Futures (fapi) order endpoint.
        Returns: (endpoint, query_string, headers)
        """
        cdef long long timestamp = int(time.time() * 1000)
        cdef list params = [
            f"symbol={symbol}",
            f"side={side}",
            f"positionSide={positionSide}",
            f"type={order_type}",
            f"quantity={quantity:.6f}".rstrip('0').rstrip('.'),
            f"newOrderRespType=RESULT",
            f"recvWindow=60000",
            f"timestamp={timestamp}"
        ]
        
        if order_type == "LIMIT" or order_type == "STOP" or order_type == "TAKE_PROFIT":
            params.append(f"price={price:.6f}".rstrip('0').rstrip('.'))
            params.append(f"timeInForce={timeInForce}")
            
        if reduceOnly:
            params.append("reduceOnly=true")
            
        cdef str query_string = "&".join(params)
        cdef str signature = self.sign_query(query_string)
        
        cdef str final_query = f"{query_string}&signature={signature}"
        
        cdef dict headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        return ("/fapi/v1/order", final_query, headers)

cpdef bytes fast_json_dump(dict data):
    """
    Bypasses GIL using orjson (which natively releases GIL).
    """
    return orjson.dumps(data)
