import asyncio
import websockets
import json
import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Warming up Numba modules
import utils.math_kernel as mk

async def warmup_loop():
    print("[WARMUP] Iniciando Pre-Calentamiento Asíncrono - JIT & PyTorch...")
    
    # 1. Calentar Numba Math Kernel (Compilación JIT de C/Rust/LLVM)
    print("[WARMUP] Compilando Numba Math Kernel con tensores de prueba...")
    start_jit = time.time()
    
    # Dummy arrays
    dummy_prices = np.random.randn(1000).astype(np.float64)
    dummy_matrix = np.random.randn(10, 100).astype(np.float64)
    
    try:
        mk.calculate_ema_jit(dummy_prices, 14)
        mk.calculate_rsi_jit(dummy_prices, 14)
        mk.calculate_macd_jit(dummy_prices, 12, 26, 9)
        mk.calculate_bollinger_jit(dummy_prices, 20, 2.0)
        mk.calculate_atr_jit(dummy_prices, dummy_prices, dummy_prices, 14)
        mk.calculate_correlation_matrix_jit(dummy_matrix)
        mk.compute_kelly_fraction_jit(0.55, 1.5, 0.45, 1.0)
        mk.fractional_differencing_jit(dummy_prices, 0.5, 1e-4)
    except Exception as e:
        print(f"[WARMUP] Warning JIT compile error: {e}")
        
    print(f"[WARMUP] JIT Cache calentado en {time.time() - start_jit:.4f}s")
    
    # 2. Calentar PyTorch
    print("[WARMUP] Calentando tensores PyTorch (CUDA/CPU)...")
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(1, 10, 10, device=device)
        y = torch.nn.functional.relu(x)
        print(f"[WARMUP] PyTorch calentado usando dispositivo: {device}")
    except ImportError:
        print("[WARMUP] PyTorch no instalado o no configurado.")
    
    # 3. Conectar a Binance WSS para simular inyección del Eje Z
    print("[WARMUP] Conectando a Binance WSS para recabar 500 ticks reales...")
    url = "wss://fstream.binance.com/ws/btcusdt@trade/ethusdt@trade/solusdt@trade"
    
    async with websockets.connect(url) as ws:
        msg_count = 0
        latencies = []
        
        while msg_count < 500:
            msg = await ws.recv()
            recv_time = time.time() * 1000
            
            data = json.loads(msg)
            # T is the timestamp of the trade on Binance servers
            if 'T' in data:
                binance_time = data['T']
                latency = recv_time - binance_time
                latencies.append(latency)
            
            msg_count += 1
            if msg_count % 100 == 0:
                print(f"[WARMUP] Ingestados {msg_count}/500 ticks...")
                
        # Report
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        min_lat = min(latencies)
        
        print("\n" + "="*50)
        print("📊 REPORTE DE LATENCIA DE RED (EJE Z - INGESTA)")
        print(f"Ticks procesados: {len(latencies)}")
        print(f"Latencia Media:   {avg_lat:.2f} ms")
        print(f"Latencia Max:     {max_lat:.2f} ms")
        print(f"Latencia Min:     {min_lat:.2f} ms")
        print("="*50)
        print("[WARMUP] PRE-FLIGHT COMPLETADO. Sistema listo para Shadow Mode.")

if __name__ == "__main__":
    asyncio.run(warmup_loop())
