# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
# core/metal/quantum_ingester.pyx
# FASE I: RETINA CUÁNTICA en Cython (Reemplazo de Rust ante ausencia de rustc)

from libc.stdlib cimport malloc, free
from libc.string cimport strchr, strncmp, memcmp
cimport cython

# Estructura compartida
cdef struct QuantumStateArena:
    size_t batch_size
    size_t num_features
    float* tensor_memory

cdef inline float _parse_float_until_quote(const unsigned char* buffer, size_t start_idx, size_t max_len, size_t* out_end_idx) nogil:
    cdef size_t i = start_idx
    cdef float val = 0.0
    cdef float sign = 1.0
    cdef float div = 10.0
    cdef bint in_fraction = False
    
    if buffer[i] == 45: # b'-'
        sign = -1.0
        i += 1

    while i < max_len and buffer[i] != 34: # b'"'
        if buffer[i] == 46: # b'.'
            in_fraction = True
        else:
            if not in_fraction:
                val = val * 10.0 + <float>(buffer[i] - 48) # 48 is '0'
            else:
                val = val + <float>(buffer[i] - 48) / div
                div *= 10.0
        i += 1
        
    out_end_idx[0] = i
    return val * sign

cdef inline bint _extract_first_level(const unsigned char* buffer, size_t buffer_len, const unsigned char* signature, size_t sig_len, float* out_price, float* out_qty) nogil:
    cdef size_t i = 0
    cdef bint found = False
    
    # Simple substring search (needle in haystack)
    while i <= buffer_len - sig_len:
        if memcmp(buffer + i, signature, sig_len) == 0:
            found = True
            break
        i += 1
        
    if not found:
        return False
        
    cdef size_t start_idx = i + sig_len
    cdef size_t end_idx = 0
    
    out_price[0] = _parse_float_until_quote(buffer, start_idx, buffer_len, &end_idx)
    
    # Saltar '","' (3 bytes)
    cdef size_t qty_start = end_idx + 3
    out_qty[0] = _parse_float_until_quote(buffer, qty_start, buffer_len, &end_idx)
    
    return True

cpdef int ingest_raw_ws_frame(size_t arena_ptr_int, const unsigned char[:] raw_bytes, size_t batch_idx) nogil:
    if arena_ptr_int == 0 or raw_bytes.shape[0] == 0:
        return 1
        
    cdef QuantumStateArena* arena = <QuantumStateArena*>arena_ptr_int
    cdef size_t length = raw_bytes.shape[0]
    cdef const unsigned char* buffer = &raw_bytes[0]
    
    cdef float bid_price = 0.0
    cdef float bid_qty = 0.0
    cdef float ask_price = 0.0
    cdef float ask_qty = 0.0
    
    cdef const unsigned char* sig_b = b'"b":[["'
    cdef const unsigned char* sig_a = b'"a":[["'
    
    _extract_first_level(buffer, length, sig_b, 7, &bid_price, &bid_qty)
    _extract_first_level(buffer, length, sig_a, 7, &ask_price, &ask_qty)
    
    cdef float total_vol = bid_qty + ask_qty + 1e-8
    cdef float imbalance = (bid_qty - ask_qty) / total_vol
    
    cdef size_t offset = batch_idx * arena.num_features
    if arena.num_features >= 10:
        arena.tensor_memory[offset + 2] = imbalance
        
    return 0

cpdef int ingest_struct_bar(size_t arena_ptr_int, size_t batch_idx, float open_p, float high_p, float low_p, float close_p, float vol) nogil:
    if arena_ptr_int == 0:
        return 1
    cdef QuantumStateArena* arena = <QuantumStateArena*>arena_ptr_int
    cdef size_t offset = batch_idx * arena.num_features
    if arena.num_features >= 5:
        arena.tensor_memory[offset + 0] = open_p
        arena.tensor_memory[offset + 1] = high_p
        arena.tensor_memory[offset + 2] = low_p
        arena.tensor_memory[offset + 3] = close_p
        arena.tensor_memory[offset + 4] = vol
    return 0

# FASE II: Muerte Digna - Watchdog ZMQ
import threading
import os
import sys

def _watchdog_loop():
    try:
        import zmq
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect("tcp://127.0.0.1:5557")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Esperar primer PING infinito para no explotar durante el boot
        sub.recv()
        
        # Imponer regla estricta de 1500ms
        sub.RCVTIMEO = 1500
        while True:
            try:
                sub.recv()
            except zmq.error.Again:
                print("☠️ [WATCHDOG INGESTER] Latido de Engine perdido por >1500ms. Aniquilación C/Rust.", file=sys.stderr)
                os._exit(1)
    except ImportError:
        pass # Ignorar si ZMQ no está disponible en este env
    except Exception as e:
        print(f"⚠️ [WATCHDOG INGESTER] Fallo: {e}")

_watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
_watchdog_thread.start()
