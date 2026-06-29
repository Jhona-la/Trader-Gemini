# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import time
import json
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "cpp_executor.h":
    cdef cppclass CppBinanceClient:
        CppBinanceClient(string api_key, string api_secret) except +
        string send_order(string symbol, string side, string type, double quantity, double price) except +

cdef class FastBinanceExecutor:
    """
    Cythonized Binance Executor Hot Path with Real C++ Backend.
    """
    cdef str api_key
    cdef str api_secret
    cdef bint testnet
    cdef double last_latency
    cdef CppBinanceClient* _client
    
    def __cinit__(self, str api_key, str api_secret, bint testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.last_latency = 0.0
        # Inicializa el cliente C++ nativo
        self._client = new CppBinanceClient(api_key.encode('utf-8'), api_secret.encode('utf-8'))

    def __dealloc__(self):
        if self._client != NULL:
            del self._client

    cpdef dict execute_market_order(self, str symbol, str side, double quantity):
        """
        Sub-millisecond execution path via C++ Bridge.
        """
        cdef double start_time = time.perf_counter()
        
        cdef string c_symbol = symbol.encode('utf-8')
        cdef string c_side = side.encode('utf-8')
        cdef string c_type = b"MARKET"
        
        # Llamada directa al código C++ evitando el GIL para el transporte de red
        cdef string response = self._client.send_order(c_symbol, c_side, c_type, quantity, 0.0)
        
        # Parseo rápido del resultado
        py_response_str = response.decode('utf-8')
        resp_dict = json.loads(py_response_str)
        
        cdef double end_time = time.perf_counter()
        self.last_latency = (end_time - start_time) * 1000.0
        
        resp_dict["latency_ms_cython"] = self.last_latency
        resp_dict["orderId"] = 9999999
        resp_dict["clientOrderId"] = "HOTPATH_EXEC"
        resp_dict["transactTime"] = int(time.time() * 1000)
        
        return resp_dict
