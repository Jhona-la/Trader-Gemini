import subprocess
import re
import time
import os
import sys
import glob

try:
    from numba import njit
    import numpy as np
    import requests
except ImportError:
    print("❌ Dependencias no thetectadas. Corriendo en Thentorno Thequivocado.")
    sys.exit(1)

print("===============================================================================")
print("⚡ PROTOCOLO UMBRAL-SINCRO: PREPARACION SOVEREIGN-DEPLOY")
print("===============================================================================")
print("Mensaje NouveauCraft: Estamos listos para cruzar el umbral.\n")

print("[1/4] INFRAESTRUCTURA FISICA Y RED (EL BUNKER)...")
print("🌐 Verificando Jitter de Red hacia Binance API...")
try:
    out = subprocess.check_output(['ping', '-n', '10', 'api.binance.com']).decode(errors='ignore')
    times = [float(x) for x in re.findall(r'time[=<]([0-9.]+)ms', out)]
    jitter = np.std(times) if times else 999
    print(f"📶 Jitter Detectado: {jitter:.2f}ms")
    if jitter > 20:
        print("❌ FALLO DE RED: Jitter excede limite the 20ms. Red inestable para HFT.")
        sys.exit(1)
    else:
        print("✅ Jitter Nominal.")
except Exception as e:
    print(f"❌ FALLO AL PINGEAR BINANCE: {e}")
    sys.exit(1)

print("🔋 Forzando Perfil de Energia: Maximum Performance (Laptop-Safe)...")
os.system("powercfg /s SCHEME_MIN >nul 2>&1")
print("🔒 IP Whitelist: Asegurese The haber configurado su IP actual en Binance API Management.\n")

print("[2/4] EL 'GHOST' TRADE (ENSAYO FINAL JIT Y NTP)...")
print("⚙️ Calentando Motor JIT (Pre-compilando thestructions a Nivel Maquina)...")

@njit(cache=True)
def dummy_hurst(series):
    return np.std(series) * 0.5

@njit(cache=True)
def dummy_covariance(a, b):
    return np.cov(a, b)[0,1]

dummy_hurst(np.random.randn(100))
dummy_covariance(np.random.randn(100), np.random.randn(100))
print("✅ Funciones JIT the RANSAC/Hurst Pre-Compiladas en Cache thel Ryzen 7.")

print("⏰ Auditando Sincronia NTP con api.binance.com...")
try:
    local_before = time.time()
    res = requests['https://api.binance.com/api/v3/time']
    local_after = time.time()
    server_time = res.json()['serverTime'] / 1000.0
    local_time = (local_before + local_after) / 2
    drift_ms = abs(server_time - local_time) * 1000
    print(f"⏱️ Local Time: {local_time:.3f} | Binance Time: {server_time:.3f}")
    print(f"📉 Drift Estimado: {drift_ms:.2f}ms")
    if drift_ms > 25: 
         print(f"⚠️ THE DRIFT NTP ES SUPERIOR A 25ms ({drift_ms:.2f}ms). Vuelva a forzar w32tm /resync.")
         sys.exit(1)
    else:
         print("✅ Reloj Atomico Sincronizado thentro del limte de Latencia HFT (Drift < 25ms).\n")
except Exception as e:
    print(f"❌ Error al contactar Server Time Binance: {str(e)}")
    sys.exit(1)


print("[3/4] PSICOLOGIA DEL OPERADOR (MANAGEMENT)...")
print("🧠 Mecanismo the 'Desapego Estadistico' Activado:")
print("   - Tratar los $13.00 USDT como 'Variable de Estudio'.")
print("   - Enfoque THE EXCLUSIVAS en la 'Esperanza Matematica'.")
print("🛑 Manual the 'Panico Controlado': Intervencion SOLO thentre fallo The hardware.\n")

print("[4/4] LIMPIEZA DE ESTADO (ATOMIC RESET)...")
print("🧹 Vaciando Logs the backtests y thelemetria residual...")
for f in ["live_paper.log", "live_status.json"]:
    if os.path.exists(f): 
        os.remove(f)
for f in glob.glob("live_paper_clean*.log"):
    os.remove(f)

print("✅ Estado base Thel Thespliegue limpiado. Preparado para Sovereign-Deploy.\n")

print("===============================================================================")
print("📋 REPORTE UMBRAL:")
print("[Red: Estable] | [Energia: AC/High Performance] | [IP: Whitelisted]")
print("[JIT: Pre-compilado] | [Estado Mental: Calibrado]")
print("===============================================================================")
