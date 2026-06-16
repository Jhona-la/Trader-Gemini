# distutils: language = c++
# cython: language_level=3

import time
cimport cython
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef struct TxState:
    int nonce
    double gas_price
    double timestamp

cdef class MempoolRbfEngine:
    """
    Zero-Copy Lock-Free C++ State Map for Mempool and RBF Tracking.
    Stores Address -> (Nonce, GasPrice, Timestamp) to detect panic updates.
    """
    # map[address] = TxState
    cdef unordered_map[string, TxState] mempool_state
    
    # Store the accumulated RBF panic score (decays exponentially)
    cdef double current_panic_score
    cdef double last_update_time
    cdef double halflife
    
    def __init__(self, halflife=10.0):
        self.current_panic_score = 0.0
        self.last_update_time = time.time()
        self.halflife = halflife
        
    cdef void _apply_decay(self, double current_time):
        """Applies exponential time decay to the global panic score."""
        cdef double age = current_time - self.last_update_time
        cdef double decay
        if age > 0:
            decay = 2.718281828459045 ** (-(0.6931471805599453 / self.halflife) * age)
            self.current_panic_score *= decay
        self.last_update_time = current_time

    def process_transaction(self, str address, int nonce, double gas_price, double gas_limit=21000.0):
        """
        Process an incoming pending transaction.
        If it's an RBF (same nonce, higher gas), calculate the Fee Delta urgency.
        Returns the calculated fee delta urgency if an RBF occurred, else 0.
        """
        cdef double current_time = time.time()
        self._apply_decay(current_time)
        
        cdef string addr_cpp = address.encode('utf-8')
        cdef double urgency = 0.0
        cdef double old_gas
        cdef double fee_delta
        cdef TxState state
        
        # Check if address exists
        if self.mempool_state.find(addr_cpp) != self.mempool_state.end():
            # Check for RBF: Same nonce, higher gas price
            if self.mempool_state[addr_cpp].nonce == nonce:
                old_gas = self.mempool_state[addr_cpp].gas_price
                if gas_price > old_gas:
                    # Fee delta in Gwei (simplified urgency score)
                    fee_delta = (gas_price - old_gas) * gas_limit
                    urgency = fee_delta
                    
                    # Accumulate to global panic score
                    self.current_panic_score += urgency
                    
            # Check if this is a newer nonce, clear state
            elif nonce > self.mempool_state[addr_cpp].nonce:
                pass # Just overwrite
                
        # Update state
        state.nonce = nonce
        state.gas_price = gas_price
        state.timestamp = current_time
        self.mempool_state[addr_cpp] = state
        
        return urgency

    def inject_mev_urgency(self, double score):
        """
        Inject an external MEV urgency score (e.g. Sandwich attack detected).
        """
        cdef double current_time = time.time()
        self._apply_decay(current_time)
        self.current_panic_score += score

    def get_panic_score(self):
        """Returns the time-decayed global mempool panic score."""
        cdef double current_time = time.time()
        self._apply_decay(current_time)
        return self.current_panic_score
        
    def get_state_size(self):
        return self.mempool_state.size()
        
    def prune_stale_transactions(self, double max_age=300.0):
        """
        Clean up memory: Remove addresses that haven't been active recently.
        Should be called periodically (e.g. every minute) from Python.
        """
        cdef double current_time = time.time()
        cdef vector[string] to_remove
        
        # Pure C++ iteration
        cdef unordered_map[string, TxState].iterator it = self.mempool_state.begin()
        while it != self.mempool_state.end():
            # In Cython, dereferencing map iterator gives a pair where first=key, second=value
            if current_time - cython.operator.dereference(it).second.timestamp > max_age:
                to_remove.push_back(cython.operator.dereference(it).first)
            cython.operator.preincrement(it)
                
        for i in range(to_remove.size()):
            self.mempool_state.erase(to_remove[i])
            
        return to_remove.size()
