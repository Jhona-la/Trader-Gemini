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


cdef class StatefulEngine:
    """
    Motor HFT O(1) para cálculo atómico de features sin bucles O(N).
    Evita Garbage Collection y mantiene el estado interno matemático.
    """
    cdef public double ema_val
    cdef public double rsi_gain
    cdef public double rsi_loss
    cdef public double last_price
    cdef public int period_ema
    cdef public int period_rsi
    cdef public bint is_initialized

    def __init__(self, int period_ema=20, int period_rsi=14):
        self.period_ema = period_ema
        self.period_rsi = period_rsi
        self.ema_val = 0.0
        self.rsi_gain = 0.0
        self.rsi_loss = 0.0
        self.last_price = 0.0
        self.is_initialized = False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def seed_history(self, cnp.ndarray[cnp.double_t, ndim=1] close_prices):
        """
        Calcula el warmup O(N) vectorizado para que el motor stateful arranque "en caliente".
        Garantiza paridad estricta con el cálculo del Backtester.
        """
        cdef int n = close_prices.shape[0]
        if n <= self.period_rsi or n <= self.period_ema:
            raise ValueError("Insufficient history for seeding (needs more candles than periods).")

        # Seeding EMA
        cdef double multiplier = 2.0 / (self.period_ema + 1.0)
        cdef double ema = close_prices[0]
        cdef int i
        
        for i in range(1, n):
            ema = (close_prices[i] - ema) * multiplier + ema
        self.ema_val = ema

        # Seeding RSI
        cdef double gain = 0.0
        cdef double loss = 0.0
        cdef double diff = 0.0

        for i in range(1, self.period_rsi + 1):
            diff = close_prices[i] - close_prices[i - 1]
            if diff > 0:
                gain += diff
            else:
                loss -= diff
                
        gain /= self.period_rsi
        loss /= self.period_rsi
        
        for i in range(self.period_rsi + 1, n):
            diff = close_prices[i] - close_prices[i - 1]
            if diff > 0:
                gain = (gain * (self.period_rsi - 1) + diff) / self.period_rsi
                loss = (loss * (self.period_rsi - 1)) / self.period_rsi
            else:
                gain = (gain * (self.period_rsi - 1)) / self.period_rsi
                loss = (loss * (self.period_rsi - 1) - diff) / self.period_rsi

        self.rsi_gain = gain
        self.rsi_loss = loss
        self.last_price = close_prices[n - 1]
        self.is_initialized = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def update_live(self, double new_price) -> tuple:
        """
        Cálculo Delta O(1) instantáneo. 
        Coste computacional mínimo. Retorna (ema, rsi) para el Ingester.
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not seeded. Call seed_history first.")

        # 1. Update EMA O(1)
        cdef double multiplier = 2.0 / (self.period_ema + 1.0)
        self.ema_val = (new_price - self.ema_val) * multiplier + self.ema_val

        # 2. Update RSI O(1)
        cdef double diff = new_price - self.last_price
        if diff > 0:
            self.rsi_gain = (self.rsi_gain * (self.period_rsi - 1) + diff) / self.period_rsi
            self.rsi_loss = (self.rsi_loss * (self.period_rsi - 1)) / self.period_rsi
        else:
            self.rsi_gain = (self.rsi_gain * (self.period_rsi - 1)) / self.period_rsi
            self.rsi_loss = (self.rsi_loss * (self.period_rsi - 1) - diff) / self.period_rsi
            
        cdef double rs = 0.0
        cdef double rsi = 100.0
        
        if self.rsi_loss != 0.0:
            rs = self.rsi_gain / self.rsi_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        self.last_price = new_price

        # En la topología SIMD final, esto usará `memcpy` directo al buffer.
        return (self.ema_val, rsi)
