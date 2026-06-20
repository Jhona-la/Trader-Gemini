import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cython_bridge.nano_ffi import NanoFFIBridge

def test_ffi():
    print("🚀 Iniciando Test de FFI Cuántico...")
    ffi = NanoFFIBridge()
    
    # Preparar Arena
    num_symbols = 5
    arena_prices = np.zeros(num_symbols, dtype=np.float32)
    arena_volumes = np.zeros(num_symbols, dtype=np.float32)
    
    # Simular mensaje WS Kline
    payload = {
        "stream": "btcusdt@kline_1m",
        "data": {
            "e": "kline",
            "E": 123456789,
            "s": "BTCUSDT",
            "k": {
                "t": 123400000,
                "T": 123459999,
                "s": "BTCUSDT",
                "i": "1m",
                "f": 100,
                "L": 200,
                "o": "60000.5",
                "c": "60150.75",
                "h": "60200.0",
                "l": "59900.0",
                "v": "15.5",
                "n": 100,
                "x": False,
                "q": "1.0",
                "V": "1.0",
                "Q": "1.0",
                "B": "0"
            }
        }
    }
    
    raw_bytes = json.dumps(payload).encode('utf-8')
    symbol_index = 2 # Simulamos que BTC es el index 2
    
    print(f"📥 Payload simulado: {len(raw_bytes)} bytes")
    
    import time
    start = time.perf_counter_ns()
    
    res = ffi.ingest_ws_frame(
        raw_bytes,
        arena_prices,
        arena_volumes,
        num_symbols,
        symbol_index
    )
    
    elapsed = time.perf_counter_ns() - start
    
    print(f"⏱️ Ingestión completada en {elapsed} ns")
    print(f"📊 Resultado FFI: {res}")
    print(f"💰 Precio actualizado en Arena[{symbol_index}]: {arena_prices[symbol_index]}")
    print(f"📦 Volumen actualizado en Arena[{symbol_index}]: {arena_volumes[symbol_index]}")
    
    if arena_prices[symbol_index] == 60150.75 and arena_volumes[symbol_index] == 15.5:
        print("✅ TEST PASSED: El FFI lee y parsea el JSON correctamente usando Rust, actualizando la Arena 10D con Zero-Copy.")
    else:
        print("❌ TEST FAILED: Los valores no coinciden.")

if __name__ == "__main__":
    test_ffi()
