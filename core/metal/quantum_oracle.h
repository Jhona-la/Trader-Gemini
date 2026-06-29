#ifndef QUANTUM_ORACLE_H
#define QUANTUM_ORACLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Alineado a 64 bytes para empatar la línea de caché (SIMD)
struct __attribute__((aligned(64))) QuantumStateArena {
    const float* prices;
    const float* volumes;
    size_t tensor_len;
    float mempool_panic_score;
    float net_liq_pressure;
    long long timestamp_ns;
};

struct TradeDecision {
    int action;
    float position_size;
    float stop_loss;
    float take_profit;
    float confidence;
    int error_code;
};

// Handshake de Topología
int validate_topological_integrity(const struct QuantumStateArena* arena, size_t stride_bytes);

// Función de Disparo (NOGIL)
void execute_oracle_kernel(const struct QuantumStateArena* arena, struct TradeDecision* out_decision);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_ORACLE_H
