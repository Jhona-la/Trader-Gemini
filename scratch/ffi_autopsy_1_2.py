import os
import sys
import time
import tracemalloc
import gc

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_bridge.quantum_bridge import CyQuantumEngine

def run_tests():
    print("🔬 INICIANDO AUTOPSIA DEL FFI: PRUEBAS 1 Y 2")
    
    # 1. Start tracemalloc
    tracemalloc.start()
    
    # Usar un engine dummy solo para leer el contador (ya que get_drop_counter no es estático)
    dummy_monitor = CyQuantumEngine()
    
    # The drop counter represents the number of ALIVE StatefulEngine instances
    alive_before = dummy_monitor.get_drop_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Motores Rust vivos iniciales: {alive_before}")
    
    def _run_engine():
        engine = CyQuantumEngine()
        
        alive_during = dummy_monitor.get_drop_counter()
        print(f"[{time.strftime('%H:%M:%S')}] Motores Rust vivos durante test: {alive_during}")
        
        # Insert 100,000 ticks
        import random
        for i in range(100_000):
            price = 1.0 + (random.random() * 0.1)
            vol = 100.0 * random.random()
            engine.process_tick(price, vol)
            
            # Get view periodically to ensure numpy doesn't leak memoryviews
            if i % 1000 == 0:
                view = engine.get_shadow_view()
                _ = view[0, 0]
                
    _run_engine()
    
    # Force garbage collection
    gc.collect()
    
    # Snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    alive_after = dummy_monitor.get_drop_counter()
    
    print("\n" + "="*50)
    print("📋 RESULTADOS DE LA AUTOPSIA FASE 2")
    print("="*50)
    print(f"PRUEBA 1: Fuga de Memoria Python-Side")
    print(f"  - Peak Memory:    {peak_mem / 1024 / 1024:.4f} MB")
    print(f"  - Current Memory: {current_mem / 1024 / 1024:.4f} MB")
    print("\n  - Top 3 Memory Allocations:")
    for stat in top_stats[:3]:
        print(f"    * {stat}")
        
    print(f"\nPRUEBA 2: Fuga de Memoria Rust-Side (Drop Audit)")
    print(f"  - Vivos inicial: {alive_before}")
    print(f"  - Vivos final:   {alive_after}")
    delta = alive_after - alive_before
    print(f"  - Fuga (Delta):  {delta} (Debe ser 0)")

if __name__ == "__main__":
    run_tests()
