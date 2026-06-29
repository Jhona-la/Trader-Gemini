import numpy as np
from numba import njit, float64
from utils.atomic_guard import jit_acquire_spinlock, jit_release_spinlock
import time

@njit(fastmath=True, cache=True, nogil=True)
def jit_consume(
    state_array: np.ndarray, # [tokens, last_update]
    cost: float,
    rate: float,
    capacity: float,
    now: float
) -> tuple:
    """
    Attempts to consume tokens.
    Returns: (success: bool, wait_time: float)
    Modifies state_array in-place if success.
    """
    tokens = state_array[0]
    last_update = state_array[1]
    
    # 1. Refill
    elapsed = now - last_update
    if elapsed > 0:
        new_tokens = tokens + (elapsed * rate)
        if new_tokens > capacity:
            new_tokens = capacity
    else:
        new_tokens = tokens
        
    # 2. Consume
    if new_tokens >= cost:
        state_array[0] = new_tokens - cost
        state_array[1] = now
        return True, 0.0
    else:
        # Calculate time to wait
        missing = cost - new_tokens
        wait_time = missing / rate
        return False, wait_time

@njit(fastmath=True, cache=True, nogil=True)
def jit_sync(
    state_array: np.ndarray,
    used_weight: float,
    capacity: float,
    rate: float,
    now: float
):
    """
    [DF-C8 FIX] Synchronize bucket with server truth.
    If server says we used X, ensuring we have at most (Capacity - X) tokens.
    """
    # 1. Update local refill first
    tokens = state_array[0]
    last_update = state_array[1]
    
    elapsed = now - last_update
    if elapsed > 0:
        tokens = min(capacity, tokens + (elapsed * rate))
    
    # 2. Server truth check
    # Server 'used' implies we have (Capacity - used) remaining.
    server_tokens = capacity - used_weight
    
    # If server says we have FEWER tokens than we think, we must downgrade.
    # If server says we have MORE, we might keep our conservative local estimate 
    # (or trust server? Safety first: take min).
    
    final_tokens = min(tokens, server_tokens)
    
    state_array[0] = final_tokens
    state_array[1] = now

class NumbaTokenBucket:
    """
    Thread-Safe, High-Precision Token Bucket.
    """
    def __init__(self, rate: float, capacity: float):
        self.rate = float(rate) # Tokens per second
        self.capacity = float(capacity)
        
        # Shared State: [tokens, last_update]
        # Initial full capacity
        self.state = np.array([capacity, time.perf_counter()], dtype=np.float64)
        
        # Spinlock
        self.lock = np.zeros(1, dtype=np.int32)
        
    def consume(self, cost: float = 1.0) -> tuple:
        """
        Thread-safe consume.
        Returns: (allowed, wait_time)
        """
        now = time.perf_counter()
        
        jit_acquire_spinlock(self.lock)
        try:
            return jit_consume(self.state, cost, self.rate, self.capacity, now)
        finally:
            jit_release_spinlock(self.lock)
            
    def predict_wait(self, cost: float) -> float:
        """
        Peek without consuming (Lock-free estimation).
        """
        now = time.perf_counter()
        # Read without lock (snapshot might be stale, but safe for prediction)
        tokens = self.state[0]
        last = self.state[1]
        
        elapsed = now - last
        curr = min(self.capacity, tokens + (elapsed * self.rate))
        
        if curr >= cost:
            return 0.0
        return (cost - curr) / self.rate

    def sync(self, used_weight: float):
        """
        [DF-C8 FIX] Sync with server headers.
        """
        now = time.perf_counter()
        jit_acquire_spinlock(self.lock)
        try:
            jit_sync(self.state, used_weight, self.capacity, self.rate, now)
        finally:
            jit_release_spinlock(self.lock)
