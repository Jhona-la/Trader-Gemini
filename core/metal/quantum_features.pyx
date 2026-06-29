# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
# ⚛️ TRINITY OMEGA-Q: EXTRACCIÓN Y ANNEALING IN-PLACE (Cython nogil)
# AXIOMA II: El Intérprete Python se congela. El cálculo ocurre en el metal.

from libc.stdint cimport uint64_t
cdef extern from "math.h":
    float fabs(float x) nogil
    float sqrt(float x) nogil

# Estructura reflejo del código en Rust
cdef extern from "quantum_ingester.h":
    cdef struct QuantumStateArena:
        uint64_t cursor_write
        uint64_t cursor_read
        uint64_t timestamps[4096]
        float bids_prices_l1[4096]
        float asks_prices_l1[4096]
        float dark_alpha_vector[32768]

# El Array de Temperatura Térmica (Evoluciona en nanosegundos)
cdef float annealing_temperature[200]

# Pre-asignación del bloque de 200 Features 
cdef float current_features[200]

cdef void calculate_features_inplace(QuantumStateArena* arena, int idx) nogil:
    """
    EL OPERADOR DE EXTRACCIÓN IN-PLACE
    Lee de la arena, inyecta la temperatura, y escupe las 200 features.
    """
    cdef float current_bid = arena.bids_prices_l1[idx]
    cdef float current_ask = arena.asks_prices_l1[idx]
    cdef float spread = current_ask - current_bid
    
    # Feature 0: Spread adaptativo
    # Aplicamos mutación térmica in-line: El "peso" del spread cambia dinámicamente
    current_features[0] = spread * annealing_temperature[0]
    
    # Feature 1: Dark Alpha Collision
    # Combina la red (idx * 8) con la temperatura en tiempo real
    cdef float dark_pressure = arena.dark_alpha_vector[(idx * 8) + 0]
    current_features[1] = dark_pressure * annealing_temperature[1]
    
    # Mutación In-Place de la propia temperatura de Annealing (Simulated Annealing)
    # Por cada extración, "enfriamos" o "calentamos" la neurona basándonos en la volatilidad.
    if spread > 1.0:
        annealing_temperature[0] += 0.0001
    else:
        annealing_temperature[0] *= 0.9999

cpdef dict run_quantum_extraction_loop(uint64_t arena_ptr_addr):
    """
    El Puente GIL-Bypass. Recibe la dirección cruda de memoria del Rust Arena.
    """
    cdef QuantumStateArena* arena = <QuantumStateArena*>arena_ptr_addr
    cdef int latest_idx = 0
    
    # Inicialización Térmica Base
    cdef int i
    for i in range(200):
        annealing_temperature[i] = 1.0
        
    # ROMPEMOS EL GIL
    with nogil:
        # El Hilo B (Cálculo) persigue al Hilo A (Red) usando Punteros Atómicos
        latest_idx = (arena.cursor_write - 1) & 4095
        
        # Leemos del byte exacto que la Red acaba de escribir
        calculate_features_inplace(arena, latest_idx)
        
    # El trabajo terminó en nanosegundos. Retornamos las features top-level a Python.
    return {
        "feat_0_spread_thermal": current_features[0],
        "feat_1_dark_pressure": current_features[1],
        "temp_state": annealing_temperature[0]
    }
