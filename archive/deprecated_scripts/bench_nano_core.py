import time
import numpy as np

try:
    from core.nano_core import calculate_unrealized_pnl_fast, NanoPriorityQueue, OrderBook
    print("[+] Nano Core cargado correctamente.")
except ImportError as e:
    print(f"[-] Error cargando Nano Core: {e}")
    exit(1)

def run_benchmark():
    # 1. PnL Fast Calculation Test
    print("\n--- Benchmarking PnL Calculation ---")
    start = time.perf_counter_ns()
    for _ in range(10000):
        calculate_unrealized_pnl_fast(50000.0, 49000.0, 1.5, "LONG")
    end = time.perf_counter_ns()
    avg_ns = (end - start) / 10000
    print(f"PnL Avg Time: {avg_ns:.2f} ns/op")
    
    # 2. Priority Queue Test
    print("\n--- Benchmarking Priority Queue ---")
    queue = NanoPriorityQueue(10000)
    start = time.perf_counter_ns()
    for i in range(10000):
        queue.put(i, 0)
    for _ in range(10000):
        queue.get()
    end = time.perf_counter_ns()
    avg_ns = (end - start) / 20000
    print(f"Queue Avg Time (Put/Get): {avg_ns:.2f} ns/op")

    # 3. OrderBook Test
    print("\n--- Benchmarking OrderBook ---")
    ob = OrderBook(max_depth=100)
    start = time.perf_counter_ns()
    for i in range(10000):
        ob.update_bid(50000.0 + i, 1.0)
    for i in range(10000):
        ob.update_bid(50000.0 + i, 0.0) # Delete
    end = time.perf_counter_ns()
    avg_ns = (end - start) / 20000
    print(f"OrderBook Avg Time (Update/Delete): {avg_ns:.2f} ns/op")

if __name__ == "__main__":
    run_benchmark()
    print("\n[V] Benchmark completado. Objetivo: < 1000 ns (1 microsegundo) por operación.")
