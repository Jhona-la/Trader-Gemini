# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import cython

cdef extern from "physics_engine.h" nogil:
    ctypedef struct PhysicsState:
        double rsi_out
        double zscore_out
        double log_return_out
        
    PhysicsState* physics_init()
    void physics_update(PhysicsState* state, double current_price)
    void physics_free(PhysicsState* state)

cdef class HyperKernel:
    cdef PhysicsState* _state

    def __cinit__(self):
        self._state = physics_init()
        if self._state is NULL:
            raise MemoryError("No se pudo reservar memoria para PhysicsState")

    def __dealloc__(self):
        if self._state is not NULL:
            physics_free(self._state)
            self._state = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple update_and_calculate(self, double current_price):
        """
        QUANTUM-NANO: O(1) in-place C-level evaluation.
        Calculates RSI(14), ZScore(20), and LogReturns without memory allocation.
        """
        # nogil block to bypass the GIL
        with nogil:
            physics_update(self._state, current_price)
        
        # Return tuple is faster than dict or array copy for 3 floats
        return (self._state.rsi_out, self._state.zscore_out, self._state.log_return_out)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple batch_update_and_calculate(self, double[:] prices):
        """
        Bulk update for warming up the indicator with historical data.
        Returns the features for the LAST element in the array.
        """
        cdef int i
        cdef int n = prices.shape[0]
        
        with nogil:
            for i in range(n):
                physics_update(self._state, prices[i])
                
        return (self._state.rsi_out, self._state.zscore_out, self._state.log_return_out)

