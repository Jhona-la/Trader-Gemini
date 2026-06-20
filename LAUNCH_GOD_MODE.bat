@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title [GOD MODE] TRADER GEMINI - OMEGA PROTOCOL
color 0E

echo ===============================================================================
echo 🛡️ GOD MODE: TRADER GEMINI - OMEGA PROTOCOL (FASE 9 + METACOGNICION)
echo ===============================================================================
echo Mensaje NouveauCraft: La automatizacion es la madre de la disciplina operativa.
echo.

:: 1. Verificacion de Red y Entorno
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ [FATAL] Environment virtual no thetectado. Lanzamiento Abortado.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

:: 2. Pre-Flight Check (Auditoria NouveauCraft y The God-Mode antigua)
echo [SYSTEM] Verificando Latencias y Precision Axioma (El Inquisidor)...
call launchers\PREFLIGHT_CHECK.bat
if %ERRORLEVEL% neq 0 (
    color 0C
    echo ❌ [FATAL] PREFLIGHT CHECK FAILED. Abortando thespliegue en OMEGA MODE.
    pause
    exit /b 1
)

:: Evaluando the Script The Python The Auditoria Antiqua Si existe
if exist "core\pre_flight.py" (
    echo [SYSTEM] Corriendo Auditoria The Modulos The HFT...
    python core/pre_flight.py
    if !ERRORLEVEL! NEQ 0 (
        color 0C
        echo ❌ [FATAL] GOD MODE MODULE AUDIT FAILED! 
        pause
        exit /b 1
    )
)

:LOOP
cls
echo ===============================================================================
echo ⚡ THE GOD ENGINE IS RUNNING (HIGH PRIORITY + OPTIMIZED)
echo ===============================================================================
echo [INFO] System: DYNAMIC ADAPTATION ENABLED + C++ ATOMICS
echo [INFO] Mode:   GOD MODE (OMNI-LATENCY AWARE)
echo [CORE] CPU Affinity: AMD Ryzen Sniper Automatic Pinning
echo [CORE] Bytecode: Optimized (-O)
echo [CORE] Buffer: Unbuffered (-u)
echo ===============================================================================

:: Start process with affinity high 
:: Para un bucle Thentro the CMD es mejor llamarlo directo thestle la consola (no `start`)
:: asi atrapamos la thetencion Thel thespliegue y lo reiniciamos.
python -O -u main.py --mode futures

echo.
echo ===============================================================================
echo ⚠️ [OMEGA ALARMA] GOD ENGINE DETENIDO.
echo 🛑 Bucle infinito desactivado para prevenir cuelgues (Flickering).
echo ===============================================================================
pause
