# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "fast_socket.cpp":
    cdef cppclass FastBinanceSocket:
        FastBinanceSocket(string key, string secret) except +
        bool connect()
        string send_order(string symbol, string side, string type, double qty, double price)
        void disconnect()

cdef class CppBinanceExecutor:
    """
    🐍🚀 CYTHON WRAPPER (Python -> C++)
    QUÉ: Clase puente entre el código Python (asyncio) y el código C++ nativo.
    POR QUÉ: Permite llamar a los sockets C++ sin la latencia de serialización JSON típica de los APIs REST de ccxt.
    """
    cdef FastBinanceSocket* _cpp_socket

    def __cinit__(self, str api_key, str secret_key):
        self._cpp_socket = new FastBinanceSocket(api_key.encode('utf-8'), secret_key.encode('utf-8'))

    def connect(self) -> bool:
        return self._cpp_socket.connect()

    def send_order_fast(self, str symbol, str side, str type, double qty, double price) -> str:
        cdef string result = self._cpp_socket.send_order(
            symbol.encode('utf-8'), 
            side.encode('utf-8'), 
            type.encode('utf-8'), 
            qty, 
            price
        )
        return result.decode('utf-8')

    def disconnect(self):
        if self._cpp_socket != NULL:
            self._cpp_socket.disconnect()

    def __dealloc__(self):
        del self._cpp_socket
