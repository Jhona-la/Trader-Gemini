import os
import sys
import time
import threading
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.quantum_bridge.quantum_bridge import CyQuantumEngine

def run_tests():
    print("🔬 INICIANDO AUTOPSIA DEL FFI: PRUEBAS 3, 4 Y 5")
    engine = CyQuantumEngine()

    print("\n" + "="*50)
    print("PRUEBA 4: Panic de Rust y Recuperación")
    print("="*50)
    # Mandar un precio normal
    success_normal = engine.process_tick(100.0, 1.0)
    print(f"Tick normal exitoso: {success_normal}")
    
    # Inyectar panic (-999.0)
    try:
        print("Enviando tick venenoso (-999.0) para invocar panic en Rust...")
        success_panic = engine.process_tick(-999.0, 1.0)
        print(f"El FFI capturó el panic con catch_unwind! Retornó success={success_panic}")
    except Exception as e:
        print(f"Excepción capturada en Python: {e}")
        
    # Verificar supervivencia
    success_post = engine.process_tick(101.0, 1.0)
    print(f"Tick posterior al panic exitoso: {success_post}")
    view = engine.get_shadow_view()
    print(f"Shadow View last_price post-panic: {view[0, 3]}")
    
    print("\n" + "="*50)
    print("PRUEBA 3 y 5: Data Race Real y Latencia bajo Contención")
    print("="*50)
    
    ITERATIONS = 100_000
    
    corrupciones = 0
    lecturas = 0
    latencias = []
    
    def writer_thread():
        import time
        nonlocal latencias
        for i in range(ITERATIONS):
            t0 = time.perf_counter_ns()
            engine.process_tick(100.0 + (i * 0.001), 1.0)
            t1 = time.perf_counter_ns()
            latencias.append(t1 - t0)
            
    def reader_thread():
        nonlocal lecturas, corrupciones
        for _ in range(ITERATIONS):
            try:
                view = engine.get_shadow_view()
                _ = view[0, 0] + view[0, 3]
                lecturas += 1
            except Exception:
                corrupciones += 1
                
    w_t = threading.Thread(target=writer_thread)
    r_t = threading.Thread(target=reader_thread)
    
    t_start = time.time()
    w_t.start()
    r_t.start()
    
    w_t.join()
    r_t.join()
    t_end = time.time()
    
    latencias = np.array(latencias) / 1000.0 # to microseconds
    p50 = np.percentile(latencias, 50)
    p95 = np.percentile(latencias, 95)
    p99 = np.percentile(latencias, 99)
    
    print(f"Iteraciones completadas: {ITERATIONS}")
    print(f"Tiempo total: {t_end - t_start:.4f}s")
    print(f"Lecturas completadas: {lecturas}")
    print(f"Corrupciones/Errores leídos: {corrupciones}")
    print(f"Latencia (µs): p50={p50:.2f}µs | p95={p95:.2f}µs | p99={p99:.2f}µs")
    
if __name__ == "__main__":
    run_tests()
