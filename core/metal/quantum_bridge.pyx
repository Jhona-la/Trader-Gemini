# cython: language_level=3, boundscheck=False, wraparound=False

from core.metal.quantum_bridge cimport QuantumStateArena, TradeDecision, validate_topological_integrity, execute_oracle_kernel

cdef public int py_validate_topological_integrity(const QuantumStateArena* arena, size_t stride_bytes):
    """
    Python-facing wrapper for handshake
    """
    return validate_topological_integrity(arena, stride_bytes)

cdef public void fire_oracle(const QuantumStateArena* arena, TradeDecision* out_decision) nogil:
    """
    MEMBRANA FFI: Libera el GIL y delega todo el poder de cómputo al Hyper-Kernel nativo.
    Se cruza una sola vez por tick de alta frecuencia.
    """
    execute_oracle_kernel(arena, out_decision)

def fire_oracle_wrapped(size_t arena_ptr, size_t decision_ptr):
    """
    Wrapper para ser llamado desde ctypes en Python pasando los punteros en crudo.
    """
    cdef QuantumStateArena* arena = <QuantumStateArena*>arena_ptr
    cdef TradeDecision* decision = <TradeDecision*>decision_ptr
    
    # Soltamos el GIL justo antes de la invocación en C/Rust
    with nogil:
        fire_oracle(arena, decision)
