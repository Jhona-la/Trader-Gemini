# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint64_t

# Ensure numpy is initialized
cnp.import_array()

cdef extern from *:
    """
    // Extern definitions expected from Rust
    typedef struct QuantumRingBuffer QuantumRingBuffer;
    typedef struct StatefulEngine StatefulEngine;

    QuantumRingBuffer* quantum_ring_new();
    void quantum_ring_free(QuantumRingBuffer* ptr);
    
    StatefulEngine* engine_new();
    void engine_free(StatefulEngine* ptr);
    
    // Returns 1 if written successfully, 0 if lap detected
    int engine_process_and_inject(StatefulEngine* engine, QuantumRingBuffer* ring, size_t reader_idx, double price, double volume);
    
    // Sandbox HFT entrypoint
    size_t run_sandbox_trial(
        const double* highs,
        const double* lows,
        const double* closes,
        const char* signals,
        size_t len,
        double sl_pct,
        double kinematic_umbral,
        double* out_pnl,
        int* out_duration,
        double* out_stats
    );
    """
    
    ctypedef struct QuantumRingBuffer:
        pass

    ctypedef struct StatefulEngine:
        pass

    QuantumRingBuffer* quantum_ring_new() nogil
    void quantum_ring_free(QuantumRingBuffer* ptr) nogil
    
    StatefulEngine* engine_new() nogil
    void engine_free(StatefulEngine* ptr) nogil
    bint engine_process_and_inject(StatefulEngine* engine, QuantumRingBuffer* ring, size_t reader_idx, double price, double volume) nogil
    size_t get_engine_drop_counter() nogil
    
    size_t run_sandbox_trial(
        const double* highs,
        const double* lows,
        const double* closes,
        const char* signals,
        size_t len,
        double sl_pct,
        double kinematic_umbral,
        double* out_pnl,
        int* out_duration,
        double* out_stats
    ) nogil

def run_sandbox_trial_py(
    cnp.ndarray[cnp.float64_t, ndim=1] highs,
    cnp.ndarray[cnp.float64_t, ndim=1] lows,
    cnp.ndarray[cnp.float64_t, ndim=1] closes,
    cnp.ndarray[cnp.int8_t, ndim=1] signals,
    double sl_pct,
    double kinematic_umbral
):
    cdef size_t n = len(closes)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_pnl = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] out_duration = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_stats = np.zeros(3, dtype=np.float64)
    
    cdef size_t trades = 0
    with nogil:
        trades = run_sandbox_trial(
            &highs[0],
            &lows[0],
            &closes[0],
            <const char*>&signals[0],
            n,
            sl_pct,
            kinematic_umbral,
            &out_pnl[0],
            <int*>&out_duration[0],
            &out_stats[0]
        )
        
    return {
        "trades": trades,
        "win_rate": out_stats[0],
        "trades_executed": out_stats[1],
        "is_pruned": out_stats[2] == 1.0,
        "pnl": out_pnl[:trades],
        "durations": out_duration[:trades]
    }

cdef class CyQuantumEngine:
    cdef QuantumRingBuffer* ring
    cdef StatefulEngine* engine
    cdef size_t current_reader_idx
    cdef size_t ring_capacity

    def __cinit__(self):
        self.ring = quantum_ring_new()
        self.engine = engine_new()
        self.current_reader_idx = 0
        self.ring_capacity = 1024

    def __dealloc__(self):
        if self.ring is not NULL:
            quantum_ring_free(self.ring)
        if self.engine is not NULL:
            engine_free(self.engine)

    cpdef bint process_tick(self, double price, double volume):
        """
        Processes a tick in Rust (f64 internal state), and injects it to the ring as f32.
        Passes current_reader_idx to the Rust Ring for Lap Detection.
        Releases the GIL for true parallelism if needed.
        """
        cdef bint success
        with nogil:
            success = engine_process_and_inject(self.engine, self.ring, self.current_reader_idx, price, volume)
        return success

    cpdef void update_reader_idx(self, size_t new_idx):
        self.current_reader_idx = new_idx

    def get_drop_counter(self):
        return get_engine_drop_counter()

    def get_shadow_view(self):
        """
        Devuelve el ndarray de numpy que es un espejo directo de la memoria de Rust (Zero-Copy).
        """
        if self.ring is NULL:
            raise RuntimeError("Engine is not initialized properly.")
        
        # El buffer del Shadow State siempre está en la memoria del anillo
        # La topología dicta que la memoria reside en C/Rust y numpy solo crea un 'memoryview'
        cdef float* ptr = <float*>self.ring
        cdef float[:, ::1] view = <float[:1, :144]>ptr
        return np.asarray(view)
