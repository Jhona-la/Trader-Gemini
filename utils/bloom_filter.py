
import math
import mmh3 # Will fall back to hashlib if not available, but let's assume standard hash first for portability
import hashlib

class BloomFilter:
    """
    🔬 PHASE 27: BLOOM FILTER
    Probabilistic data structure for O(1) set membership testing.
    Used for high-speed deduplication (e.g., have we seen this trade ID?).
    
    Space efficient: ~10MB can store millions of items with < 1% false positive rate.
    """
    
    def __init__(self, capacity=100000, error_rate=0.01):
        """
        capacity: anticipated number of elements
        error_rate: acceptable probability of false positives
        """
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Optimal number of bits (m)
        self.num_bits = int(- (capacity * math.log(error_rate)) / (math.log(2) ** 2))
        
        # Optimal number of hash functions (k)
        self.num_hashes = int((self.num_bits / capacity) * math.log(2))
        
        # Bit array (using bytearray for efficiency)
        self.size_bytes = (self.num_bits + 7) // 8
        self.bit_array = bytearray(self.size_bytes)
        
        self.count = 0

    def _get_hashes(self, item):
        """Generates k hash values for the item."""
        # We use double hashing to simulate k hash functions
        # hash(i) = (hash1 + i * hash2) % num_bits
        item_str = str(item).encode('utf-8')
        
        # Hash 1: SHA256
        h1 = int(hashlib.sha256(item_str).hexdigest(), 16)
        
        # Hash 2: MD5
        h2 = int(hashlib.md5(item_str).hexdigest(), 16)
        
        hashes = []
        for i in range(self.num_hashes):
            combined = (h1 + i * h2) % self.num_bits
            hashes.append(combined)
            
        return hashes

    def add(self, item):
        """Adds an item to the filter."""
        for digest in self._get_hashes(item):
            byte_index = digest // 8
            bit_index = digest % 8
            self.bit_array[byte_index] |= (1 << bit_index)
        self.count += 1

    def __contains__(self, item):
        """Checks if item is likely in the set (True) or definitely not (False)."""
        for digest in self._get_hashes(item):
            byte_index = digest // 8
            bit_index = digest % 8
            if not (self.bit_array[byte_index] & (1 << bit_index)):
                return False # Definitely not present
        return True # Probably present

    def __len__(self):
        return self.count
