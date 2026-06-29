# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython

cpdef np.ndarray[np.float64_t, ndim=1] calc_atr_metal(np.ndarray[np.float64_t, ndim=1] high, 
                                                      np.ndarray[np.float64_t, ndim=1] low, 
                                                      np.ndarray[np.float64_t, ndim=1] close, 
                                                      int period):
    cdef int n = high.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] atr = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] tr = np.empty(n, dtype=np.float64)
    cdef int i
    
    if n == 0:
        return atr
        
    tr[0] = high[0] - low[0]
    atr[0] = tr[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr

cpdef np.ndarray[np.float64_t, ndim=1] calc_rsi_metal(np.ndarray[np.float64_t, ndim=1] close, 
                                                      int period):
    cdef int n = close.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] rsi = np.empty(n, dtype=np.float64)
    cdef int i
    cdef double delta
    cdef double up = 0.0
    cdef double down = 0.0
    cdef double rs
    
    if n == 0:
        return rsi
        
    rsi[0] = 0.0
    
    # Calculate first smoothed averages
    for i in range(1, min(period + 1, n)):
        delta = close[i] - close[i-1]
        if delta > 0:
            up += delta
        else:
            down -= delta
    
    up /= period
    down /= period
    
    if down == 0:
        if i < n:
            rsi[i] = 100.0
    else:
        if i < n:
            rs = up / down
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    for i in range(period + 1, n):
        delta = close[i] - close[i-1]
        if delta > 0:
            up = (up * (period - 1) + delta) / period
            down = (down * (period - 1)) / period
        else:
            up = (up * (period - 1)) / period
            down = (down * (period - 1) - delta) / period
            
        if down == 0:
            rsi[i] = 100.0
        else:
            rs = up / down
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


cpdef double vpin_score_multiplier(double vpin, int signal_direction, int flow_direction):
    if vpin > 0.7 and signal_direction == flow_direction:
        return 1.15
    elif vpin < 0.3:
        return 0.85
    return 1.0

cpdef np.ndarray[np.float64_t, ndim=1] calc_vpin_metal(np.ndarray[np.float64_t, ndim=1] buy_volume,
                                                       np.ndarray[np.float64_t, ndim=1] sell_volume,
                                                       int period):
    cdef int n = buy_volume.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] vpin = np.empty(n, dtype=np.float64)
    cdef int i
    cdef double sum_buy = 0.0
    cdef double sum_sell = 0.0
    cdef double total_vol = 0.0

    if n == 0:
        return vpin

    for i in range(n):
        sum_buy += buy_volume[i]
        sum_sell += sell_volume[i]

        if i >= period:
            sum_buy -= buy_volume[i - period]
            sum_sell -= sell_volume[i - period]

        total_vol = sum_buy + sum_sell

        if total_vol > 0 and i >= period - 1:
            vpin[i] = abs(sum_buy - sum_sell) / total_vol
        else:
            vpin[i] = 0.0

    return vpin
