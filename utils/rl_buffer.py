import numpy as np
from numba import njit, int64, float32, float64
from utils.atomic_guard import jit_acquire_spinlock, jit_release_spinlock

@njit(fastmath=True, cache=True)
def jit_push(
    head: int,
    size: int,
    capacity: int,
    s_buffer: np.ndarray,
    a_buffer: np.ndarray,
    r_buffer: np.ndarray,
    sn_buffer: np.ndarray,
    s: np.ndarray,
    a: float,
    r: float,
    sn: np.ndarray
) -> tuple:
    """
    Pushes a transition (s, a, r, sn) into the ring buffer.
    Returns (new_head, new_size)
    """
    # Overwrite at head
    s_buffer[head] = s
    a_buffer[head] = a
    r_buffer[head] = r
    sn_buffer[head] = sn
    
    new_head = (head + 1) % capacity
    new_size = min(size + 1, capacity)
    
    return new_head, new_size

@njit(fastmath=True, cache=True)
def jit_sample(
    size: int,
    batch_size: int,
    s_buffer: np.ndarray,
    a_buffer: np.ndarray,
    r_buffer: np.ndarray,
    sn_buffer: np.ndarray
) -> tuple:
    """
    Samples a random batch from the valid range [0, size).
    Returns tuple of (batch_s, batch_a, batch_r, batch_sn)
    """
    if size < 1:
        # Return empty/zeros if empty (should be handled by caller)
        # We can't easily return empty with correct shape without alloc
        # Accessing empty buffer is UB/Logic error usually.
        return (
            np.zeros((batch_size, s_buffer.shape[1]), dtype=float32),
            np.zeros(batch_size, dtype=float32),
            np.zeros(batch_size, dtype=float32),
            np.zeros((batch_size, sn_buffer.shape[1]), dtype=float32)
        )
        
    # Generate random indices
    indices = np.random.randint(0, size, batch_size)
    
    # Alloc outputs
    # Note: allocating inside jit is fast enough for small batches
    batch_s = np.empty((batch_size, s_buffer.shape[1]), dtype=float32)
    batch_a = np.empty(batch_size, dtype=float32)
    batch_r = np.empty(batch_size, dtype=float32)
    batch_sn = np.empty((batch_size, sn_buffer.shape[1]), dtype=float32)
    
    for i in range(batch_size):
        idx = indices[i]
        batch_s[i] = s_buffer[idx]
        batch_a[i] = a_buffer[idx]
        batch_r[i] = r_buffer[idx]
        batch_sn[i] = sn_buffer[idx]
        
    return batch_s, batch_a, batch_r, batch_sn

class NumbaExperienceBuffer:
    """
    High-Performance Memory Buffer for RL Agents using Numba.
    - Zero copy storage.
    - O(1) Push/Sample JIT speed.
    - Thread-Safe via AtomicGuard.
    """
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.head = 0
        self.size = 0
        
        # Pre-allocate contiguous memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        
        # Spinlock State
        from utils.atomic_guard import AtomicGuard
        self.guard = AtomicGuard()
        
    
    def push(self, s, a, r, sn):
        """Append transition (Thread-Safe)"""
        # Spin-Lock
        self.guard.acquire()
        try:
            self.head, self.size = jit_push(
                self.head, self.size, self.capacity,
                self.states, self.actions, self.rewards, self.next_states,
                s.astype(np.float32), float(a), float(r), sn.astype(np.float32)
            )
        finally:
            self.guard.release()
        
    def sample(self, batch_size: int):
        """Get batch (Thread-Safe Reading)"""
        self.guard.acquire()
        try:
            real_batch = min(batch_size, self.size)
            return jit_sample(
                self.size, real_batch,
                self.states, self.actions, self.rewards, self.next_states
            )
        finally:
            self.guard.release()

    def __len__(self):
        return self.size
