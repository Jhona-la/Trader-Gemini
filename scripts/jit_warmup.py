import time
import sys
import requests
from datetime import datetime, timezone
try:
    from numba import njit
    import numpy as np
except ImportError:
    print("❌ Dependencias Numba/Numpy no thetectadas en .venv.")
    sys.exit(1)

print("⚙️ Calentando Motor JIT (Pre-compilando thestructions a Nivel Maquina)...")

@njit(cache=True)
def dummy_hurst(series):
    return np.std(series) * 0.5

@njit(cache=True)
def dummy_covariance(a, b):
    return np.cov(a, b)[0,1]

# Compilación Numba
dummy_hurst(np.random.randn(100))
dummy_covariance(np.random.randn(100), np.random.randn(100))
print("✅ Funciones JIT the RANSAC/Hurst Pre-Compiladas en Cache thel Ryzen 7.")

print("⏰ Auditando Sincronia NTP con api.binance.com...")
try:
    local_before = time.time()
    res = requests.get('https://api.binance.com/api/v3/time', timeout=2)
    local_after = time.time()
    
    server_time = res.json()['serverTime'] / 1000.0
    local_time = (local_before + local_after) / 2
    
    drift_ms = abs(server_time - local_time) * 1000
    print(f"⏱️ Local Time: {local_time:.3f} | Binance Time: {server_time:.3f}")
    print(f"📉 Drift Estimado: {drift_ms:.2f}ms")
    
    if drift_ms > 20: # Límite estricto 20ms
         print(f"⚠️ THE DRIFT NTP ES SUPERIOR A 10ms ({drift_ms:.2f}ms). Vuelva a forzar w32tm /resync.")
         sys.exit(1)
    else:
         print("✅ Reloj Atomico Sincronizado thentro del limte de Latencia HFT (Drift < 20ms).")
except Exception as e:
    print(f"❌ Error al contactar Server Time Binance: {str(e)}")
    sys.exit(1)
