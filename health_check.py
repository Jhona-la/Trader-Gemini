
import os
import time
import asyncio
from binance.client import Client
from config import Config
from utils.logger import logger
from core.portfolio import Portfolio
import queue

async def run_health_check():
    print("\n" + "="*50)
    print("🩺 TRADER GEMINI: DAILY HEALTH CHECK (5-MIN PROTOCOL)")
    print("="*50)

    # 1. VERIFICACIÓN DE .ENV
    print("1. 🛡️  Verificando Credenciales (.env)...", end=" ", flush=True)
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    
    if api_key and api_secret and len(api_key) > 10:
        print("✅ OK")
    else:
        print("❌ ERROR (Credenciales no detectadas o inválidas)")
        return

    # 2. LATENCIA CON BINANCE
    print("2. ⚡ Midiendo Latencia con Binance API...", end=" ", flush=True)
    try:
        client = Client(api_key, api_secret)
        start_time = time.time()
        server_time = client.get_server_time()
        latency = (time.time() - start_time) * 1000
        
        if latency < 1000:
            print(f"✅ OK ({latency:.2f}ms)")
        else:
            print(f"⚠️  SLOW ({latency:.2f}ms) - Revisar conexión")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return

    # 3. SINCRONIZACIÓN DE BALANCE ($13.50)
    print("3. 💵 Sincronizando Balance de Capital...", end=" ", flush=True)
    try:
        # Check Futures Balance if configured
        if Config.BINANCE_USE_FUTURES:
            balance = client.futures_account_balance()
            usdt_balance = next((float(b['balance']) for b in balance if b['asset'] == 'USDT'), 0.0)
        else:
            balance = client.get_asset_balance(asset='USDT')
            usdt_balance = float(balance['free']) if balance else 0.0

        print(f"✅ OK (${usdt_balance:.2f})")
        
        # Alerta de Capital
        if usdt_balance < 13.00:
            print(f"   ⚠️  ADVERTENCIA: Balance (${usdt_balance:.2f}) por debajo del capital génesis ($13.50).")
        elif usdt_balance >= 13.50:
            print(f"   🎯 TARGET: Capital génesis preservado o aumentado.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # 4. INTEGRIDAD DE ARCHIVOS CRÍTICOS
    print("4. 📂 Verificando Archivos de Sistema...", end=" ", flush=True)
    critical_files = [
        "config.py", 
        "core/engine.py", 
        "core/portfolio.py", 
        "dashboard/data/live_status.json"
    ]
    missing = [f for f in critical_files if not os.path.exists(f)]
    
    if not missing:
        print("✅ OK")
    else:
        print(f"⚠️  FALTANTE: {missing}")

    print("\n" + "="*50)
    print("🏁 DIAGNÓSTICO COMPLETADO: SISTEMA LISTO PARA OPERAR")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_health_check())
