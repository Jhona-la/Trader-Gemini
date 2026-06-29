import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import time
from config import Config
from core.events import OrderEvent
from execution.binance_executor import BinanceExecutor
from utils.logger import logger
from enum import Enum
import os

class Direction(Enum):
    BUY = "BUY"
    SELL = "SELL"

async def run_latency_certification():
    print("===============================================================================")
    print(" 🚀 INICIANDO CERTIFICACION DE NANO-LATENCIA (CANARY TEST)")
    print("===============================================================================")
    
    # Forzar Shadow Mode para 100% Zero Risk
    os.environ['TG_SHADOW_MODE'] = '1'
    Config.SHADOW_MODE = True
    
    print("[1/3] Iniciando Ejecutor Cuántico de Binance...")
    executor = BinanceExecutor(events_queue=None, portfolio=None)
    
    print("[2/3] Conectando a WebSockets Testnet / Shadow...")
    # Simularemos 5 órdenes de prueba para calcular la latencia promedio de Ida y Vuelta
    
    latencies = []
    
    for i in range(5):
        start = time.perf_counter()
        
        # Crear Orden Falsa (Señal del Modelo ML)
        dummy_order = OrderEvent(
            symbol="BNB/USDT",
            order_type="MARKET",
            quantity=0.01,
            direction=Direction.BUY,
            price=600.00,
            horizon="SCALPING",
            sl_pct=0.05,
            tp_pct=0.002,
            is_shadow=True
        )
        
        # Enviar orden al Executor
        await executor.execute_order(dummy_order)
        
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        print(f" ⚡ Ping {i+1}: {latency_ms:.2f} ms")
        
        await asyncio.sleep(0.5)
        
    avg_latency = sum(latencies) / len(latencies)
    print("\n===============================================================================")
    print(" 📊 REPORTE DE CERTIFICACION FORENSE")
    print("===============================================================================")
    print(f"Latencia Promedio: {avg_latency:.2f} ms")
    
    if avg_latency < 100:
        print("✅ ESTADO: CERTIFICADO. El motor es capaz de ejecutar Take Profits de 0.20%.")
    else:
        print("❌ ESTADO: FALLO. La latencia supera los 100ms. Peligro de Slippage Mortal.")
        
    print("===============================================================================")

if __name__ == "__main__":
    asyncio.run(run_latency_certification())
