import numpy as np
import random
from collections import deque
from typing import Tuple, List, Any
import pickle
import os

from utils.logger import logger

class SumTree:
    """
    SumTree structure for efficient priority sampling.
    Leaf nodes store priorities. Internal nodes store sum of children.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree array size: 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Data array stores the actual experiences
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: Any):
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Stores experiences and samples them based on priority (TD-error).
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01  # Small constant to ensure non-zero priority

    def _get_priority(self, error: float) -> float:
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error: float, experience: Tuple, is_black_swan: bool = False):
        """
        Add a new experience to the buffer.
        experience: (state, action, reward, next_state, done)
        
        [PHASE 9] Black Swan Priority:
        If event is critical (High Volatility/Crash), we boost priority to Max.
        """
        p = self._get_priority(error)
        
        if is_black_swan:
            # Boost to max priority (force replay)
            # Find max priority currently in tree or default to 10.0
            max_p = np.max(self.tree.tree[-self.capacity:]) 
            if max_p <= 0: max_p = 1.0
            p = max_p * 2.0 # Double max priority
            
        self.tree.add(p, experience)

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], List[float]]:
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, p, data) = self.tree.get(s)
            
            # Verify data exists (handle edge case where tree might be slightly inconsistent)
            if data is None: 
                # Fallback: recover by picking a random valid entry if possible
                if self.tree.n_entries > 0:
                    idx = random.randint(0, self.tree.n_entries - 1) + self.capacity - 1
                    p = self.tree.tree[idx]
                    data = self.tree.data[idx - self.capacity + 1]
                else:
                    continue

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx: int, error: float):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
    def __len__(self):
        return self.tree.n_entries

    def save(self, filepath: str):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'tree': self.tree,
                    'beta': self.beta
                }, f)
            logger.info(f"Memory buffer saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save memory buffer: {e}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            logger.warning(f"Memory buffer file {filepath} not found. Starting fresh.")
            return
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.tree = data['tree']
                self.beta = data['beta']
            logger.info(f"Memory buffer loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load memory buffer: {e}")
