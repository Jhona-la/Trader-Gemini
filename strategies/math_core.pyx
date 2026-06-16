# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as cnp

# Ensure NumPy types are initialized
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double fast_sma(cnp.ndarray[cnp.double_t, ndim=1] data, int window):
    """
    Computes SMA using native C loop. (~10-20ns for small arrays)
    """
    cdef int n = data.shape[0]
    cdef double s = 0.0
    cdef int i
    
    if window <= 0:
        return 0.0
    if n < window:
        window = n
        
    for i in range(n - window, n):
        s += data[i]
        
    return s / window

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double fast_ema(cnp.ndarray[cnp.double_t, ndim=1] data, int window):
    """
    Computes EMA natively.
    """
    cdef int n = data.shape[0]
    if n == 0 or window <= 0:
        return 0.0
    
    cdef double multiplier = 2.0 / (window + 1.0)
    cdef double ema = data[0]
    cdef int i
    
    for i in range(1, n):
        ema = (data[i] - ema) * multiplier + ema
        
    return ema

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple fast_rsi(cnp.ndarray[cnp.double_t, ndim=1] close_prices, int window):
    """
    Returns (rsi_value, current_gain, current_loss) natively.
    """
    cdef int n = close_prices.shape[0]
    if n <= window or window <= 0:
        return (50.0, 0.0, 0.0)
        
    cdef double gain = 0.0
    cdef double loss = 0.0
    cdef double diff = 0.0
    cdef int i
    
    # Simple average for the first 'window' periods
    for i in range(1, window + 1):
        diff = close_prices[i] - close_prices[i - 1]
        if diff > 0:
            gain += diff
        else:
            loss -= diff
            
    gain /= window
    loss /= window
    
    # Wilder's Smoothing for the rest
    for i in range(window + 1, n):
        diff = close_prices[i] - close_prices[i - 1]
        if diff > 0:
            gain = (gain * (window - 1) + diff) / window
            loss = (loss * (window - 1)) / window
        else:
            gain = (gain * (window - 1)) / window
            loss = (loss * (window - 1) - diff) / window
            
    cdef double rs = 0.0
    if loss == 0.0:
        return (100.0, gain, loss)
        
    rs = gain / loss
    cdef double rsi = 100.0 - (100.0 / (1.0 + rs))
    return (rsi, gain, loss)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double fast_std(cnp.ndarray[cnp.double_t, ndim=1] data, int window):
    """
    Computes standard deviation natively.
    """
    cdef int n = data.shape[0]
    if n < window:
        window = n
    if window <= 1:
        return 0.0
        
    cdef double mean = 0.0
    cdef double sq_diff_sum = 0.0
    cdef int i
    
    for i in range(n - window, n):
        mean += data[i]
    mean /= window
    
    for i in range(n - window, n):
        sq_diff_sum += (data[i] - mean) ** 2
        
    return (sq_diff_sum / window) ** 0.5
