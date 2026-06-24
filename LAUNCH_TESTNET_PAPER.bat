@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title [TESTNET MODE] TRADER GEMINI - MODO PAPER
color 0E

echo ===============================================================================
echo  TESTNET MODE: TRADER GEMINI - MODO PAPER (FASE VI)
echo ===============================================================================
echo Mensaje NouveauCraft: Ejecucion Directa O(1) hacia Binance Testnet Futures.
echo.

:: 1. Verificacion de Red y Entorno
if not exist ".venv\Scripts\activate.bat" (
    echo  [FATAL] Environment virtual no detectado. Lanzamiento Abortado.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

:: 2. Pre-Flight Check
echo [SYSTEM] Verificando Latencias...
if exist "launchers\PREFLIGHT_CHECK.bat" (
    call launchers\PREFLIGHT_CHECK.bat
    if !ERRORLEVEL! neq 0 (
        color 0C
        echo  [FATAL] PREFLIGHT CHECK FAILED. Abortando despliegue.
        pause
        exit /b 1
    )
)

:LOOP
cls
echo ===============================================================================
echo  THE TESTNET ENGINE IS RUNNING
echo ===============================================================================
echo [INFO] System: PAPER MODE (REAL EXECUTION ON TESTNET)
echo [INFO] Mode:   MODO PRUEBA DE PAPEL (BYPASS ACTIVO)
echo [CORE] BINANCE_USE_TESTNET=True
echo ===============================================================================

:: Forzar uso de Futures Demo
set BINANCE_USE_DEMO=True
set BINANCE_USE_TESTNET=False
set TG_SHADOW_MODE=0

:: Filtrar unicamente monedas de alta liquidez para prevenir ruido en la simulacion
set TG_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT

python -O -u main.py --mode futures

echo.
echo ===============================================================================
echo  [OMEGA ALARMA] TESTNET ENGINE DETENIDO.
echo ===============================================================================
pause
