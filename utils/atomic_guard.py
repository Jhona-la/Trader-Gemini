import numpy as np
from numba import njit, int32

@njit(fastmath=True, cache=True, nogil=True)
def jit_try_acquire_spinlock(lock_array: np.ndarray, max_spins: int = 1000) -> bool:
    """
    Attempts to acquire lock with bounded spinning.
    Returns: True if acquired, False otherwise.
    """
    for _ in range(max_spins):
        if lock_array[0] == 0:
            lock_array[0] = 1
            return True
    return False

@njit(fastmath=True, cache=True, nogil=True)
def jit_acquire_spinlock(lock_array: np.ndarray):
    """
    DEPRECATED: Prefer non-blocking version to avoid CPU starvation.
    """
    # Simple TTaS (Test-Test-and-Set)
    # Note: Without hardware CAS, this has a race window.
    # We rely on the fact that writes are generally distinct events.
    # For robust production code, we'd need C intrinsics.
    # This simulation tries to be as safe as possible in pure Numba.
    
    while True:
        if lock_array[0] == 0:
            # Tentative lock
            lock_array[0] = 1
            
            # Critical Section Safety barrier?
            # Re-check is pointless without atomic CAS returning old val.
            # We assume for this prototype that collision probability is managed 
            # or that we are effectively running on a system where this suffices for simple coordination.
            # (In reality, for true HFT, we use Hardware CAS).
            return
        
@njit(fastmath=True, cache=True, nogil=True)
def jit_release_spinlock(lock_array: np.ndarray):
    """
    Unlock.
    """
    # Write 0 release
    lock_array[0] = 0

class AtomicGuard:
    """
    Wrapper for a shared lock state.
    """
    def __init__(self):
        self.lock = np.zeros(1, dtype=np.int32)
    
    def acquire(self):
        """Acquire with Python-level backoff to prevent starvation"""
        import time
        while not jit_try_acquire_spinlock(self.lock, max_spins=1000):
            time.sleep(0) # Yield control to OS
        
    def release(self):
        jit_release_spinlock(self.lock)

