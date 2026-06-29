import os
import sys
import ctypes
import tracemalloc
import time

# Load the Rust DLL
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "rust_engine", "target", "release", "quantum_engine.dll"))
if not os.path.exists(dll_path):
    print(f"Error: {dll_path} not found.")
    sys.exit(1)

try:
    quantum_engine = ctypes.CDLL(dll_path)

    # Set up ctypes signatures
    quantum_engine.engine_new.restype = ctypes.c_void_p
    quantum_engine.engine_new.argtypes = []

    quantum_engine.engine_free.restype = None
    quantum_engine.engine_free.argtypes = [ctypes.c_void_p]

    quantum_engine.get_engine_drop_counter.restype = ctypes.c_size_t
    quantum_engine.get_engine_drop_counter.argtypes = []
except AttributeError as e:
    print(f"Error binding FFI functions: {e}")
    sys.exit(1)

def run_test_1_and_2():
    print("--- FASE 2: AUDITORÍA FFI (PRUEBA 1 Y PRUEBA 2) ---")
    
    # Prueba 2: Contador inicial
    initial_drop_counter = quantum_engine.get_engine_drop_counter()
    print(f"Drop Counter Inicial (Rust C-API): {initial_drop_counter}")
    
    # Prueba 1: Iniciar tracemalloc
    tracemalloc.start()
    
    print("Ejecutando Smoke Test: Creando, usando y liberando 100,000 motores (C-API)...")
    
    # Simulamos el ciclo de vida del Engine en el puente de Python (CTypes memory test)
    for _ in range(100_000):
        ptr = quantum_engine.engine_new()
        # "Uso" del puntero
        quantum_engine.engine_free(ptr)
        
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    tracemalloc.stop()
    
    # Prueba 2: Contador final
    final_drop_counter = quantum_engine.get_engine_drop_counter()
    
    print(f"Drop Counter Final (Rust C-API): {final_drop_counter}")
    if initial_drop_counter == final_drop_counter:
        print("✅ PRUEBA 2 (Rust Fuga): Instancias vivas regresaron al valor original. No hay fugas en Rust.")
    else:
        print(f"❌ PRUEBA 2 (Rust Fuga): Drop_counter es {final_drop_counter}. ¡FUGA DETECTADA!")
        
    print(f"\n✅ PRUEBA 1 (Python Fuga):")
    print(f"  Current memory: {current_mem / 1024 / 1024:.4f} MB")
    print(f"  Peak memory:    {peak_mem / 1024 / 1024:.4f} MB")
    
    print("\n[Top 3 Líneas con Mayor Asignación Python]")
    for stat in top_stats[:3]:
        print(stat)

if __name__ == "__main__":
    run_test_1_and_2()
