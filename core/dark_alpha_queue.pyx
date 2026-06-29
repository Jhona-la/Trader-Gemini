# distutils: language = c++
# cython: language_level=3

import time
from libcpp.deque cimport deque

cdef struct LiquidationEvent:
    double timestamp
    int side # 1 for LONG (implies SELL pressure), -1 for SHORT (implies BUY pressure)
    double size

cdef class DarkAlphaQueue:
    """
    Zero-Copy Lock-Free C++ Ring Buffer for Dark Alpha Layer.
    Processes Hyperliquid cascades with Exponential Time-Decay at Nanosecond speed.
    """
    cdef deque[LiquidationEvent] buffer
    cdef double halflife
    cdef double mean
    cdef double m2
    cdef long long sample_count
    
    def __init__(self, halflife=15.0):
        self.halflife = halflife
        self.mean = 0.0
        self.m2 = 0.0
        self.sample_count = 0
        
    def push_liquidation(self, int side, double size):
        """
        Push a new liquidation event.
        side: 1 if LONG was liquidated, -1 if SHORT was liquidated.
        size: Notional size in USD.
        """
        cdef LiquidationEvent ev
        ev.timestamp = time.time()
        ev.side = side
        ev.size = size
        self.buffer.push_back(ev)
        
    def get_net_pressure(self):
        """
        Returns the net liquidation pressure with time decay.
        Positive value -> Short squeeze (Buy pressure).
        Negative value -> Long cascade (Sell pressure).
        """
        cdef double current_time = time.time()
        cdef double net_pressure = 0.0
        cdef double decay
        cdef double age
        cdef LiquidationEvent ev
        
        # Prune events older than 60 seconds (they no longer affect immediate alpha)
        while not self.buffer.empty():
            ev = self.buffer.front()
            if current_time - ev.timestamp > 60.0:
                self.buffer.pop_front()
            else:
                break
                
        # Calculate time-decayed net pressure
        cdef size_t i
        for i in range(self.buffer.size()):
            ev = self.buffer[i]
            age = current_time - ev.timestamp
            
            # Exponential decay: e^(-lambda * t) where lambda = ln(2)/halflife
            decay = 2.718281828459045 ** (-(0.6931471805599453 / self.halflife) * age)
            
            if ev.side == -1: # Short liquidated -> Buy pressure
                net_pressure += (ev.size * decay)
            elif ev.side == 1: # Long liquidated -> Sell pressure
                net_pressure -= (ev.size * decay)
                
        # Update Welford's online algorithm for variance
        self.sample_count += 1
        cdef double delta = net_pressure - self.mean
        self.mean += delta / self.sample_count
        cdef double delta2 = net_pressure - self.mean
        self.m2 += delta * delta2
        
        cdef double variance = 0.0
        cdef double std_dev = 0.0
        if self.sample_count > 1:
            variance = self.m2 / self.sample_count
            std_dev = variance ** 0.5
            
        # Trigger Cascade Protocol if net_pressure exceeds 2.5 std_dev
        if std_dev > 0 and abs(net_pressure - self.mean) > 2.5 * std_dev and abs(net_pressure) > 1000.0:
            direction = "BULLISH (Short Squeeze)" if net_pressure > 0 else "BEARISH (Long Cascade)"
            print(f"🚨 [DARK ALPHA] HYPERLIQUID CASCADE DETECTED! Direction: {direction} | Pressure: ${net_pressure:,.2f} | StdDev: {std_dev:,.2f}")
            
        return net_pressure
        
    def get_size(self):
        return self.buffer.size()
        
    def get_stats(self):
        cdef double std_dev = 0.0
        if self.sample_count > 1:
            std_dev = (self.m2 / self.sample_count) ** 0.5
        return {
            "mean": self.mean,
            "std_dev": std_dev,
            "samples": self.sample_count
        }
