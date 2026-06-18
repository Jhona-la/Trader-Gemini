import sys
import time
import asyncio
from core.engine import PriorityBoundedQueue

class MockEvent:
    def __init__(self, type_name, priority):
        self.type = type_name
        self.priority = priority

async def benchmark_queue():
    print("🚀 Iniciando Benchmark de NanoPriorityQueue (Cython C-RingBuffer)...")
    
    queue = PriorityBoundedQueue(maxsize=1000000)
    
    # Check if cython is loaded
    is_nano = getattr(queue, '_use_cython', False)
    print(f"✅ NanoPriorityQueue Activado: {is_nano}")
    
    # 1. Burst Injection Test
    num_events = 500000
    print(f"\n⚡ Inyectando {num_events} Ticks de mercado en ráfaga (Burst Mode)...")
    start_time = time.perf_counter()
    
    for i in range(num_events):
        # priority 0 = Fill/Critical
        queue.put(MockEvent('FILL', 0))
        
    inject_time = time.perf_counter() - start_time
    print(f"⏱️ Tiempo de Inyección: {inject_time:.6f} segundos")
    print(f"⚡ Velocidad de Inyección: {num_events / inject_time:,.0f} Ticks/segundo")
    print(f"🔬 Latencia por Tick (Put): {(inject_time / num_events) * 1e9:.2f} nanosegundos")
    
    # 2. Burst Consumption Test
    print(f"\n🔥 Devorando {num_events} Ticks...")
    start_time = time.perf_counter()
    
    consumed = 0
    while queue._items_count > 0:
        event = await queue.get()
        consumed += 1
        
    consume_time = time.perf_counter() - start_time
    print(f"⏱️ Tiempo de Consumo: {consume_time:.6f} segundos")
    print(f"🔥 Velocidad de Consumo: {consumed / consume_time:,.0f} Ticks/segundo")
    print(f"🔬 Latencia por Tick (Get): {(consume_time / consumed) * 1e9:.2f} nanosegundos")
    print("\n✅ Benchmark Completado Exitosamente!")

if __name__ == "__main__":
    asyncio.run(benchmark_queue())
