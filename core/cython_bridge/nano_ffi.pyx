# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
cimport cython
from libc.stdint cimport uint64_t, int64_t, int32_t, uint8_t

cdef extern from "stdint.h":
    pass

# Import the Rust FFI bindings
cdef extern from *:
    """
    // Minimal C declarations matching the Rust #[repr(C)] structs
    #include <stdint.h>
    #include <stdbool.h>
    
    typedef struct {
        const float* prices;
        const float* volumes;
        size_t tensor_len;
        float mempool_panic_score;
        float net_liq_pressure;
        int64_t timestamp_ns;
    } QuantumStateArena;

    typedef struct {
        int32_t action;
        float position_size;
        float stop_loss;
        float take_profit;
        float confidence;
        int32_t error_code;
        float mempool_panic;
        float net_liq_pressure;
        float liquidation_cascade;
    } TradeDecision;
    
    typedef struct {
        uint64_t seqlock;
        float data[1024][144];
        uint64_t lap_violation_count;
    } QuantumRingBuffer;
    
    // Rust functions
    QuantumRingBuffer* quantum_ring_new();
    void quantum_ring_free(QuantumRingBuffer* ptr);
    
    // C-FFI entry point from Quantum Arena (needs implementation in Rust side)
    TradeDecision execute_oracle_kernel(const QuantumStateArena* arena);
    
    typedef struct {
        double stop_price;
        bool force_close;
        int32_t new_phase;
        double max_pnl_pct;
        double mfe_atr;
    } TrailingResult;

    TrailingResult evaluate_quantum_trailing(
        int32_t pos_side,
        double entry_price,
        double current_price,
        double current_atr,
        int32_t current_phase,
        double mfe_atr,
        double max_pnl_pct,
        double current_trail_stop,
        double pullback_tol,
        double trail_f1,
        double trail_f2,
        double trail_f3,
        double trail_runner
    );

    int32_t ingest_raw_ws_frame(
        QuantumStateArena* arena_ptr,
        const uint8_t* raw_bytes_ptr,
        size_t length,
        size_t index
    );
    """
    
    ctypedef struct QuantumStateArena:
        const float* prices
        const float* volumes
        size_t tensor_len
        float mempool_panic_score
        float net_liq_pressure
        int64_t timestamp_ns

    ctypedef struct TradeDecision:
        int32_t action
        float position_size
        float stop_loss
        float take_profit
        float confidence
        int32_t error_code
        float mempool_panic
        float net_liq_pressure
        float liquidation_cascade
        
    ctypedef struct QuantumRingBuffer:
        uint64_t seqlock
        float data[1024][144]
        uint64_t lap_violation_count

    ctypedef struct TrailingResult:
        double stop_price
        bint force_close
        int32_t new_phase
        double max_pnl_pct
        double mfe_atr

    QuantumRingBuffer* quantum_ring_new() nogil
    void quantum_ring_free(QuantumRingBuffer* ptr) nogil
    TradeDecision execute_oracle_kernel(const QuantumStateArena* arena) nogil
    TrailingResult evaluate_quantum_trailing(
        int32_t pos_side,
        double entry_price,
        double current_price,
        double current_atr,
        int32_t current_phase,
        double mfe_atr,
        double max_pnl_pct,
        double current_trail_stop,
        double pullback_tol,
        double trail_f1,
        double trail_f2,
        double trail_f3,
        double trail_runner
    ) nogil
    int32_t ingest_raw_ws_frame(
        QuantumStateArena* arena_ptr,
        const uint8_t* raw_bytes_ptr,
        size_t length,
        size_t index
    ) nogil

cdef class NanoFFIBridge:
    """
    Zero-Copy Cython Bridge.
    Maps Python Numpy/MemoryViews directly to C pointers to bypass GIL and allocations.
    """
    cdef QuantumRingBuffer* _ring_ptr

    def __cinit__(self):
        self._ring_ptr = quantum_ring_new()
        if self._ring_ptr is NULL:
            raise MemoryError("Failed to allocate QuantumRingBuffer in Rust")

    def __dealloc__(self):
        if self._ring_ptr is not NULL:
            quantum_ring_free(self._ring_ptr)
            self._ring_ptr = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def invoke_oracle(self, float[::1] prices, float[::1] volumes, float mempool_panic, float net_liq, int64_t timestamp) -> tuple:
        """
        Nogil execution path for the Oracle Kernel.
        """
        cdef QuantumStateArena arena
        arena.prices = &prices[0]
        arena.volumes = &volumes[0]
        arena.tensor_len = prices.shape[0]
        arena.mempool_panic_score = mempool_panic
        arena.net_liq_pressure = net_liq
        arena.timestamp_ns = timestamp

        cdef TradeDecision decision
        
        # Release the GIL and invoke Rust Core
        with nogil:
            decision = execute_oracle_kernel(&arena)
            
        return (
            decision.action, 
            decision.position_size, 
            decision.stop_loss, 
            decision.take_profit, 
            decision.confidence,
            decision.error_code,
            decision.mempool_panic,
            decision.net_liq_pressure,
            decision.liquidation_cascade
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_trailing(self, int pos_side, double entry_price, double current_price, double current_atr, int current_phase, double mfe_atr, double max_pnl_pct, double current_trail_stop, double pullback_tol, double trail_f1, double trail_f2, double trail_f3, double trail_runner) -> dict:
        """
        Nogil execution for Quantum Trailing Stops.
        """
        cdef TrailingResult res
        
        with nogil:
            res = evaluate_quantum_trailing(
                pos_side,
                entry_price,
                current_price,
                current_atr,
                current_phase,
                mfe_atr,
                max_pnl_pct,
                current_trail_stop,
                pullback_tol,
                trail_f1,
                trail_f2,
                trail_f3,
                trail_runner
            )
            
        return {
            'stop_price': res.stop_price,
            'force_close': bool(res.force_close),
            'new_phase': res.new_phase,
            'max_pnl_pct': res.max_pnl_pct,
            'mfe_atr': res.mfe_atr
        }

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ingest_ws_frame(self, const unsigned char[::1] raw_bytes, float[::1] prices, float[::1] volumes, size_t tensor_len, size_t index) -> int32_t:
        """
        Nogil Zero-Copy WebSockets JSON Parser.
        Deserializes a raw byte slice into the memory mapped arena directly.
        """
        cdef QuantumStateArena arena
        arena.prices = &prices[0]
        arena.volumes = &volumes[0]
        arena.tensor_len = tensor_len
        arena.mempool_panic_score = 0.0
        arena.net_liq_pressure = 0.0
        arena.timestamp_ns = 0

        cdef int32_t result
        
        with nogil:
            result = ingest_raw_ws_frame(
                &arena,
                <const uint8_t*>&raw_bytes[0],
                raw_bytes.shape[0],
                index
            )
            
        return result
