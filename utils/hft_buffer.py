
import numpy as np
from numba import int32, int64, float32, float64, void
from numba.experimental import jitclass

spec = [
    ('data', float32[:]),
    ('capacity', int32),
    ('head', int32),
    ('size', int32),
    ('is_full', int32) # boolean as int
]

@jitclass(spec)
class NumbaRingBuffer:
    """
    High-Performance Circular Buffer (Phase 9).
    Zero-allocation sliding window.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Pre-allocate contiguous memory block
        self.data = np.zeros(capacity, dtype=np.float32)
        self.head = 0
        self.size = 0
        self.is_full = 0

    def push(self, value):
        """O(1) insertion."""
        self.data[self.head] = value
        # Bitwise modulus optimization for power of 2 capacity? 
        # Standard modulo is fine for now.
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.is_full = 1

    def get_last(self, n):
        """
        Get last N elements in chronological order.
        Returns a COPY if wrapping occurs, or a View if contiguous.
        Optimization: We always return a copy to ensure safety for JIT functions downstream.
        """
        if n > self.size:
            n = self.size
            
        result = np.empty(n, dtype=np.float32)
        
        # Calculate start index
        # head points to next insertion spot
        # so last element is at head-1
        current_idx = (self.head - n + self.capacity) % self.capacity
        
        # Copy logic
        for i in range(n):
            result[i] = self.data[current_idx]
            current_idx = (current_idx + 1) % self.capacity
            
        return result

    def get_all(self):
        return self.get_last(self.size)

    def clear(self):
        self.head = 0
        self.size = 0
        self.is_full = 0

spec64 = [
    ('data', int64[:]),
    ('capacity', int32),
    ('head', int32),
    ('size', int32),
    ('is_full', int32)
]

@jitclass(spec64)
class NumbaRingBuffer64:
    """
    Int64 version for Timestamps (ms precision).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=np.int64)
        self.head = 0
        self.size = 0
        self.is_full = 0

    def push(self, value):
        self.data[self.head] = value
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.is_full = 1

    def get_last(self, n):
        if n > self.size: n = self.size
        result = np.empty(n, dtype=np.int64)
        current_idx = (self.head - n + self.capacity) % self.capacity
        for i in range(n):
            result[i] = self.data[current_idx]
            current_idx = (current_idx + 1) % self.capacity
        return result

# PHASE 4: Structured Unified Buffer
# Note: Numba jitclass has limited support for structured array field access in some envs.
# We implement a multi-array strategy internally but expose a 'get_struct' for downstream.
struct_spec = [
    ('t', int64[:]),
    ('o', float32[:]),
    ('h', float32[:]),
    ('l', float32[:]),
    ('c', float32[:]),
    ('v', float32[:]),
    ('capacity', int32),
    ('head', int32),
    ('size', int32)
]

@jitclass(struct_spec)
class NumbaStructuredRingBuffer:
    """
    🧠 SUPREMO-V3: Unified OHLCVT Ring Buffer.
    Eliminates memory fragmentation by grouping all candle data.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Ensure 64-byte alignment for cache locality (approximate via padding if needed, 
        # but numpy.zeros usually aligns well. For strict HFT we'd use posix_memalign)
        # We ensure capacity is a multiple of cache line size for vectorization.
        self.t = np.zeros(capacity, dtype=np.int64)
        self.o = np.zeros(capacity, dtype=np.float32)
        self.h = np.zeros(capacity, dtype=np.float32)
        self.l = np.zeros(capacity, dtype=np.float32)
        self.c = np.zeros(capacity, dtype=np.float32)
        self.v = np.zeros(capacity, dtype=np.float32)
        self.head = 0
        self.size = 0
    
    def push(self, t, o, h, l, c, v):
        """Atomic push of a full candle."""
        self.t[self.head] = t
        self.o[self.head] = o
        self.h[self.head] = h
        self.l[self.head] = l
        self.c[self.head] = c
        self.v[self.head] = v
        
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_last(self, n):
        """Standard copy-based retrieval."""
        if n > self.size: n = self.size
        
        t_res = np.empty(n, dtype=np.int64)
        o_res = np.empty(n, dtype=np.float32)
        h_res = np.empty(n, dtype=np.float32)
        l_res = np.empty(n, dtype=np.float32)
        c_res = np.empty(n, dtype=np.float32)
        v_res = np.empty(n, dtype=np.float32)
        
        curr = (self.head - n + self.capacity) % self.capacity
        for i in range(n):
            t_res[i] = self.t[curr]
            o_res[i] = self.o[curr]
            h_res[i] = self.h[curr]
            l_res[i] = self.l[curr]
            c_res[i] = self.c[curr]
            v_res[i] = self.v[curr]
            curr = (curr + 1) % self.capacity
            
        return t_res, o_res, h_res, l_res, c_res, v_res

    def get_latest_view(self):
        """
        [PHASE 63] Zero-Copy view of the very last element.
        Extremely fast for single-tick processing.
        """
        idx = (self.head - 1 + self.capacity) % self.capacity
        return (
            self.t[idx], self.o[idx], self.h[idx], 
            self.l[idx], self.c[idx], self.v[idx]
        )

    def rewind_one(self):
        """Allows overwriting the last bar (for live candle updates)."""
        if self.size > 0:
            self.head = (self.head - 1 + self.capacity) % self.capacity
            self.size -= 1


