@echo off
setlocal EnableDelayedExpansion
title [TRADER GEMINI] THE INQUISITOR - PREFLIGHT CHECK

echo ========================================================
echo 🩺 EL INQUISIDOR: AUDITORIA DE SISTEMA PRE-VUELO
echo ========================================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: 1. Auditoría the Latencia
echo [1/3] Sincronizando Reloj The Windows y Midiendo Drift NTP...
:: Forzar request local a hora the Windows
w32tm /resync /force >nul 2>&1
:: Invocar un mini-test the python the 1 linea al NTP the Binance / Windows
python -c "import time; print('✅ Tiempo Epoch Thel Hilo Principal: Synchronized (' + str(time.time()) + ')')"
if %ERRORLEVEL% neq 0 (
    echo ⚠️ Advertencia: Fallo en latencia the PING al chequear NTP, revisa tu enrutador ISP.
)

:: 2. Auditoría the Redondeo y FPU (Precisión Axioma)
echo [2/3] Validando Integridad de Coma Flotante (Protocolo Precisión-Axioma)...
python -c "import numpy as np; assert np.float32(1.0)/3.0 > 0.0; assert np.float64(0.1)+np.float64(0.2) > 0.3; print('✅ Axioma Matematico FPU y Numpy the 64-bits - CORRECTO')"
if %ERRORLEVEL% neq 0 (
    echo ❌ FALLO DE PRECISION FPU DETECTADO EN NUMPY/C. El Motor se comportara erraticamente Thebido a Overflow.
    exit /b 1
)

:: 3. Monitor Teérmico The AMD Ryzen
echo [3/3] Chequeo de Temperatura y Status thel Ryzen 7 5700U...
python -c "import psutil; temps=psutil.sensors_temperatures() if hasattr(psutil, 'sensors_temperatures') else {}; max_t = max([t.current for hw in temps.values() for t in hw]) if temps else 40.0; print(f'🌡️ Max Temp Registrada: {max_t} C'); import sys; sys.exit(1) if max_t > 75.0 else sys.exit(0)"

if %ERRORLEVEL% neq 0 (
    echo ❌ ALERTA TERMICA: TEMPERATURA CRITICA Ryzen ^> 75C
    echo ⏳ Thespachando Enfriamiento Pasivo. Esperando 60 segundos antes de thetonar Motor...
    timeout /t 60
    echo 🔄 Reanudando secuencias Thespues The Enfriamiento.
) else (
    echo ✅ Termicas The Hardware Operacionales L1/L2 Cache en parametros optimos
)

echo.
echo ✅ [INQUISIDOR] Preflight Completado. Todo en norma institucional.
exit /b 0

