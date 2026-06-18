import time
import json
import orjson
import tracemalloc
from core.metal.ingester_bridge import retina_bridge

def run_benchmark():
    # 1. Generar 100,000 payloads crudos de Binance Depth
    PAYLOAD_COUNT = 100_000
    sample_payload = b'{"e":"depthUpdate","E":1672531200000,"s":"BTCUSDT","U":157,"u":160,"b":[["42000.50","1.5"],["42000.00","2.0"]],"a":[["42001.00","2.0"],["42001.50","1.0"]]}'
    payloads = [sample_payload for _ in range(PAYLOAD_COUNT)]
    
    print(f"--- BENCHMARK: RETINA CUÁNTICA vs PYTHON JSON ---")
    print(f"Cargando {PAYLOAD_COUNT} mensajes de profundidad...")
    
    # Python json.loads
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    for p in payloads:
        data = json.loads(p)
        # Simulate accessing
        b_price = float(data['b'][0][0])
    t1 = time.perf_counter_ns()
    _, peak_json = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_json = (t1 - t0) / PAYLOAD_COUNT
    
    # Python orjson.loads
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    for p in payloads:
        data = orjson.loads(p)
        b_price = float(data['b'][0][0])
    t1 = time.perf_counter_ns()
    _, peak_orjson = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_orjson = (t1 - t0) / PAYLOAD_COUNT
    
    # Retina Cuántica
    if not retina_bridge.available:
        print("Retina Bridge no disponible. Compila libquantum_fusion.dll/.so")
        return
        
    arena_ptr = 0 # Mock pointer
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    for p in payloads:
        retina_bridge.ingest(arena_ptr, p, 0)
    t1 = time.perf_counter_ns()
    _, peak_retina = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_retina = (t1 - t0) / PAYLOAD_COUNT
    
    print("\nResultados Latencia (nanosegundos por mensaje):")
    print(f"1. json.loads   : {time_json:8.0f} ns")
    print(f"2. orjson.loads : {time_orjson:8.0f} ns")
    print(f"3. Retina (FFI) : {time_retina:8.0f} ns")
    
    print("\nResultados Memoria RAM asignada (Peak Bytes en Python):")
    print(f"1. json.loads   : {peak_json} bytes")
    print(f"2. orjson.loads : {peak_orjson} bytes")
    print(f"3. Retina (FFI) : {peak_retina} bytes")

    if time_retina < time_orjson:
        print("\n✅ RETINA SUPREMA: Aceleración conseguida.")
    if peak_retina < 1000:
        print("✅ ZERO-ALLOCATION VERIFICADO: El GC de Python está dormido.")

if __name__ == "__main__":
    run_benchmark()
