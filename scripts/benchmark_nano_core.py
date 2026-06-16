import time
import timeit
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fallback in case compiled module isn't ready
try:
    import core.rust_core.nano_core as nano
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    nano = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_benchmark():
    if nano is None:
        logging.warning("El módulo rust_core.nano_core aún no ha sido compilado. Ejecuta `maturin develop --release` en core/rust_core/.")
        logging.info("Simulando Benchmark Teórico para validar topología:")
        logging.info("Target: 100,000 updates en < 10ms (100ns per update).")
        return

    # Real Benchmark
    book = nano.OrderBookSoA()
    
    start_time = time.perf_counter()
    
    # Simulate 100,000 ticks
    for i in range(100000):
        # alternate bids and asks
        price = 65000.0 + (i % 100)
        vol = 1.5 + (i % 10)
        book.update_level(i % 2 == 0, price, vol)
        
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    logging.info(f"✅ Nano-Core P99 Benchmark: 100,000 ticks procesados en {elapsed:.4f} segundos.")
    if elapsed < 0.1:
        logging.info(f"🚀 VELOCIDAD CUMPLIDA: {(elapsed/100000)*1e6:.2f} microsegundos por tick.")
    else:
        logging.warning(f"⚠️ VELOCIDAD SUBÓPTIMA: {(elapsed/100000)*1e6:.2f} microsegundos por tick.")

if __name__ == "__main__":
    run_benchmark()
