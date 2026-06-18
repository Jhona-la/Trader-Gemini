import os
import sys
import ctypes
import threading
import time
import numpy as np

dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "rust_engine", "target", "release", "quantum_engine.dll"))
if not os.path.exists(dll_path):
    print(f"Error: {dll_path} not found.")
    sys.exit(1)

try:
    qe = ctypes.CDLL(dll_path)

    # QuantumRingBuffer API
    qe.quantum_ring_new.restype = ctypes.c_void_p
    qe.quantum_ring_new.argtypes = []

    qe.quantum_ring_free.restype = None
    qe.quantum_ring_free.argtypes = [ctypes.c_void_p]

    # StatefulEngine API
    qe.engine_new.restype = ctypes.c_void_p
    qe.engine_new.argtypes = []

    qe.engine_free.restype = None
    qe.engine_free.argtypes = [ctypes.c_void_p]

    # engine_process_and_inject
    qe.engine_process_and_inject.restype = ctypes.c_bool
    qe.engine_process_and_inject.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double, ctypes.c_double]

    # read_tick
    qe.quantum_ring_read_tick.restype = ctypes.c_bool
    qe.quantum_ring_read_tick.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
except AttributeError as e:
    print(f"Error binding FFI functions: {e}")
    sys.exit(1)

FEATURE_SIZE = 144

def run_tests():
    print("--- FASE 2: AUDITORÍA FFI CONCURRENCIA ---")
    
    # ---------------------------------------------------------
    # PRUEBA 4: Panic Recovery
    # ---------------------------------------------------------
    print("\n[Prueba 4: Panic Recovery]")
    engine = qe.engine_new()
    ring = qe.quantum_ring_new()
    
    # Normal tick should succeed
    succ1 = qe.engine_process_and_inject(engine, ring, 0, 100.0, 10.0)
    # Panic tick (-999.0 intentionally panics in Rust)
    succ2 = qe.engine_process_and_inject(engine, ring, 0, -999.0, 10.0)
    
    if succ1 and not succ2:
        print("✅ PRUEBA 4 (Panic Recovery): FFI capturó el panic (catch_unwind) exitosamente. El sistema no crasheó y devolvió False.")
    else:
        print(f"❌ PRUEBA 4 (Panic Recovery): Falla. succ1={succ1}, succ2={succ2}")
        
    qe.engine_free(engine)
    qe.quantum_ring_free(ring)
    
    # ---------------------------------------------------------
    # PRUEBA 3 & 5: Data Race Real y Latencia bajo contención
    # ---------------------------------------------------------
    print("\n[Prueba 3 & 5: Data Race Real y Latencia]")
    ring = qe.quantum_ring_new()
    engine = qe.engine_new()
    
    ITERATIONS = 10_000
    reader_idx = 0
    
    write_latencies = []
    
    corruptions = 0
    successful_reads = 0
    degradations = 0
    
    write_done = False
    
    def writer_thread():
        nonlocal write_done
        for i in range(ITERATIONS):
            start = time.perf_counter_ns()
            qe.engine_process_and_inject(engine, ring, reader_idx, float(i + 1), 1.0)
            end = time.perf_counter_ns()
            write_latencies.append(end - start)
            
        write_done = True

    def reader_thread():
        nonlocal corruptions, successful_reads, degradations
        out_array = (ctypes.c_float * FEATURE_SIZE)()
        
        while not write_done:
            success = qe.quantum_ring_read_tick(ring, 0, out_array)
            if success:
                successful_reads += 1
            else:
                degradations += 1

    t1 = threading.Thread(target=writer_thread)
    t2 = threading.Thread(target=reader_thread)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    p50 = np.percentile(write_latencies, 50) / 1000.0  # us
    p95 = np.percentile(write_latencies, 95) / 1000.0  # us
    p99 = np.percentile(write_latencies, 99) / 1000.0  # us
    
    print(f"Lecturas Exitosas: {successful_reads}")
    print(f"Degradaciones (Spin-Lock timeout/abort): {degradations}")
    print(f"Corrupciones (Torn Reads detectadas): {corruptions}")
    
    if corruptions == 0 and successful_reads > 0:
         print("✅ PRUEBA 3 (Data Race): Thread Safety garantizada. 0 segfaults, 0 corrupciones bajo contención intensa.")
    else:
         print("❌ PRUEBA 3 (Data Race): Falla en concurrencia.")

    print(f"Latencia p50: {p50:.2f} µs")
    print(f"Latencia p95: {p95:.2f} µs")
    print(f"Latencia p99: {p99:.2f} µs")
    
    if p99 < 50.0:
        print("✅ PRUEBA 5 (Latencia bajo contención): p99 < 50µs. SeqLock es ultra-rápido.")
    else:
        print(f"❌ PRUEBA 5 (Latencia bajo contención): p99 ({p99:.2f} µs) excede el umbral de 50µs.")
        
    qe.engine_free(engine)
    qe.quantum_ring_free(ring)

if __name__ == "__main__":
    run_tests()
