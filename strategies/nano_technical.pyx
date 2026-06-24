# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import cython
from libc.stdlib cimport malloc, free

cdef struct SignalOutput:
    double strength
    int signal_type  # 1 for LONG, -1 for SHORT, 0 for HOLD

cdef class NanoTechnicalStrategy:
    """
    Zero-Copy Native Technical Strategy Generator.
    Replaces dictionary allocations and Python loops with raw C-structs.
    """
    cdef double strength_threshold
    cdef double adx_threshold

    def __init__(self, double strength_threshold=0.35, double adx_threshold=20.0):
        self.strength_threshold = strength_threshold
        self.adx_threshold = adx_threshold

    @cython.cdivision(True)
    cdef SignalOutput compute_signal(self, double* close, double* high, double* low, int length, double current_adx, double current_rsi):
        cdef SignalOutput out
        out.strength = 0.0
        out.signal_type = 0
        
        if length < 50:
            return out
            
        cdef double c = close[length - 1]
        
        # Super simplified native condition just to prove the pipeline
        if current_adx > self.adx_threshold:
            if current_rsi < 30.0:
                out.signal_type = 1
                out.strength = 0.8
            elif current_rsi > 70.0:
                out.signal_type = -1
                out.strength = 0.8
                
        if out.strength < self.strength_threshold:
            out.signal_type = 0
            
        return out

    cpdef dict generate_signal_wrapper(self, list closes, double adx, double rsi):
        """ Python bridge for testing """
        cdef int length = len(closes)
        cdef double* c_closes = <double*>malloc(length * sizeof(double))
        
        cdef int i
        for i in range(length):
            c_closes[i] = closes[i]
            
        cdef SignalOutput res = self.compute_signal(c_closes, NULL, NULL, length, adx, rsi)
        free(c_closes)
        
        return {
            "signal_type": "LONG" if res.signal_type == 1 else "SHORT" if res.signal_type == -1 else "HOLD",
            "strength": res.strength
        }
