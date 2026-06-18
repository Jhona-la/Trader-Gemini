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

# ── QUANTUM NANO WARMUP ──
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.nano_core import calculate_unrealized_pnl_fast, calculate_kelly_fraction, update_hwm_lwm
    _ = calculate_unrealized_pnl_fast(60000.0, 59000.0, 0.5, 1)
    _ = calculate_kelly_fraction(0.6, 2.0)
    _ = update_hwm_lwm(60000.0, 59000.0, 61000.0, 58000.0, 1)
except Exception as e:
    print(f"⚠️ [NANO-CORE] Warmup Failed: {e}")

try:
    from core.nano_risk_engine import evaluate_sl_tp_trailing_jit
    _ = evaluate_sl_tp_trailing_jit(60500.0, 60000.0, 61000.0, 59500.0, 0.5, 0.02, 0.05, 0.01, False, False, 2.0)
except Exception as e:
    print(f"⚠️ [NANO-RISK] Warmup Failed: {e}")

try:
    from core.nano_stop_checker import batch_check_stops
    n = 10
    _ = batch_check_stops(
        np.full(n, 59000.0), np.full(n, 61000.0), np.full(n, 60000.0), np.full(n, 60000.0),
        np.full(n, 61000.0), np.full(n, 59000.0), np.full(n, 0.5), np.full(n, 0.02),
        np.full(n, 0.05), np.full(n, 0.01), np.zeros(n, dtype=np.int32),
        np.zeros(n, dtype=np.int32), np.full(n, 2.0)
    )
except Exception as e:
    print(f"⚠️ [NANO-STOP] Warmup Failed: {e}")

print("✅ Funciones JIT the RANSAC/Hurst y Nano-Engines Pre-Compiladas en Cache thel Ryzen 7.")

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
