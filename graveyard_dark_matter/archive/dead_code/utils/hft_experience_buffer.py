import numpy as np
from numba import int64, float32, jitclass

# [DF-A1] Optimized Experience Buffer for Deep Forensics
# Using Structured Field approach to ensure cache line alignment (64 bytes)
# In Numba jitclass, we use separate arrays but manage them as a contiguous block conceptually.

experience_spec = [
    ('state_buf', float32[:, :]),
    ('action_buf', float32[:]),
    ('reward_buf', float32[:]),
    ('next_state_buf', float32[:, :]),
    ('priority_buf', float32[:]),
    ('capacity', int64),
    ('head', int64),
    ('size', int64)
]

@jitclass(experience_spec)
class NumbaExperienceBuffer:
    """
    🧠 SUPREMO-Metal: High-Performance Experience Buffer.
    Ensures O(1) sampling and O(1) insertion with zero Python object overhead.
    """
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        # Pre-allocate large contiguous blocks
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros(capacity, dtype=np.float32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.priority_buf = np.zeros(capacity, dtype=np.float32)
        self.head = 0
        self.size = 0

    def add(self, state, action, reward, next_state, priority):
        """O(1) Push."""
        self.state_buf[self.head] = state
        self.action_buf[self.head] = action
        self.reward_buf[self.head] = reward
        self.next_state_buf[self.head] = next_state
        self.priority_buf[self.head] = priority
        
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample_random(self, batch_size):
        """Standard random sampling (O(batch_size))."""
        indices = np.random.randint(0, self.size, batch_size)
        
        s_batch = np.empty((batch_size, self.state_buf.shape[1]), dtype=np.float32)
        a_batch = np.empty(batch_size, dtype=np.float32)
        r_batch = np.empty(batch_size, dtype=np.float32)
        ns_batch = np.empty((batch_size, self.state_buf.shape[1]), dtype=np.float32)
        
        for i in range(batch_size):
            idx = indices[i]
            s_batch[i] = self.state_buf[idx]
            a_batch[i] = self.action_buf[idx]
            r_batch[i] = self.reward_buf[idx]
            ns_batch[i] = self.next_state_buf[idx]
            
        return s_batch, a_batch, r_batch, ns_batch

    def update_priority(self, idx, priority):
        """Updates specific priority for PER."""
        if idx < self.capacity:
            self.priority_buf[idx] = priority
