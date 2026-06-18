# cython: language_level=3

cdef extern from "quantum_oracle.h":
    cdef struct QuantumStateArena:
        const float* prices
        const float* volumes
        size_t tensor_len
        float mempool_panic_score
        float net_liq_pressure
        long long timestamp_ns

    cdef struct TradeDecision:
        int action
        float position_size
        float stop_loss
        float take_profit
        float confidence
        int error_code

    int validate_topological_integrity(const QuantumStateArena* arena, size_t stride_bytes)
    void execute_oracle_kernel(const QuantumStateArena* arena, TradeDecision* out_decision) nogil
