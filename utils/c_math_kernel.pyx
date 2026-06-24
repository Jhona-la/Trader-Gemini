# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, log, fabs, fmax

# ==============================================================================
# 🧠 FASE 1: CYTHONIZACIÓN DE ESTRATEGIAS MATEMÁTICAS (NANO-SPEED)
# ==============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_kahan_sum(double[:] arr) nogil:
    cdef double sum_val = 0.0
    cdef double c = 0.0
    cdef double y, t
    cdef Py_ssize_t i, n = arr.shape[0]
    
    for i in range(n):
        y = arr[i] - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=1] calculate_ema_cython(double[:] prices, int period):
    cdef int n = prices.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ema = np.full(n, np.nan, dtype=np.float64)
    cdef double alpha = 2.0 / (period + 1.0)
    cdef double sma = 0.0
    cdef int i
    
    if n < period:
        return ema
        
    for i in range(period):
        sma += prices[i]
    
    ema[period-1] = sma / period
    
    for i in range(period, n):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
        
    return ema

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=1] calculate_rsi_cython(double[:] prices, int period):
    cdef int n = prices.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rsi = np.full(n, np.nan, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] deltas = np.zeros(n, dtype=np.float64)
    
    if n <= period:
        return rsi
        
    cdef int i
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i-1]
        
    cdef double gain = 0.0
    cdef double loss = 0.0
    cdef double d
    
    for i in range(1, period + 1):
        d = deltas[i]
        if d > 0:
            gain += d
        else:
            loss -= d
            
    cdef double avg_gain = gain / period
    cdef double avg_loss = loss / period
    cdef double rs
    
    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        
    for i in range(period + 1, n):
        d = deltas[i]
        if d > 0:
            avg_gain = (avg_gain * (period - 1) + d) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - d) / period
            
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple calculate_bollinger_cython(double[:] prices, int period, double std_dev):
    cdef int n = prices.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] upper = np.full(n, np.nan, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] middle = np.full(n, np.nan, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lower = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower
        
    cdef double window_sum = 0.0
    cdef double window_sum_sq = 0.0
    cdef double val_new, val_old, mean, variance, std
    cdef int i
    
    for i in range(period):
        window_sum += prices[i]
        window_sum_sq += prices[i] * prices[i]
        
    for i in range(period - 1, n):
        if i >= period:
            val_new = prices[i]
            val_old = prices[i - period]
            window_sum += val_new - val_old
            window_sum_sq += val_new * val_new - val_old * val_old
            
        mean = window_sum / period
        variance = (window_sum_sq / period) - (mean * mean)
        
        if variance < 0.0:
            variance = 0.0
            
        std = sqrt(variance)
        middle[i] = mean
        upper[i] = mean + (std * std_dev)
        lower[i] = mean - (std * std_dev)
        
    return upper, middle, lower

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=1] calculate_atr_cython(double[:] high, double[:] low, double[:] close, int period):
    cdef int n = close.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tr = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] atr = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return atr
        
    cdef int i
    cdef double h_l, h_pc, l_pc
    
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_pc = fabs(high[i] - close[i-1])
        l_pc = fabs(low[i] - close[i-1])
        tr[i] = fmax(h_l, fmax(h_pc, l_pc))
        
    cdef double tr_sum = 0.0
    for i in range(1, period + 1):
        tr_sum += tr[i]
        
    atr[period] = tr_sum / period
    
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    return atr
