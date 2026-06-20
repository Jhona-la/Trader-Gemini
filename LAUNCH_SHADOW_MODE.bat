@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title [SHADOW MODE] TRADER GEMINI - MODO SOMBRA
color 0B

echo ===============================================================================
echo ?? SHADOW MODE: TRADER GEMINI - MODO SOMBRA (FASE III)
echo ===============================================================================
echo Mensaje NouveauCraft: La simulacion hiper-realista es la unica manera de 
echo medir el Espejo Cuantico sin sangrar capital.
echo.

:: 1. Verificacion de Red y Entorno
if not exist ".venv\Scripts\activate.bat" (
    echo ? [FATAL] Environment virtual no detectado. Lanzamiento Abortado.
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
        echo ? [FATAL] PREFLIGHT CHECK FAILED. Abortando despliegue.
        pause
        exit /b 1
    )
)

:: Evaluando el Script de Python
if exist "core\pre_flight.py" (
    echo [SYSTEM] Corriendo Auditoria The Modulos The HFT...
    python core/pre_flight.py
    if !ERRORLEVEL! NEQ 0 (
        color 0C
        echo ? [FATAL] PRE FLIGHT MODULE AUDIT FAILED! 
        pause
        exit /b 1
    )
)

:LOOP
cls
echo ===============================================================================
echo ?? THE SHADOW ENGINE IS RUNNING
echo ===============================================================================
echo [INFO] System: SHADOW MODE (VIRTUAL EXECUTION)
echo [INFO] Mode:   MODO SOMBRA CUANTICO
echo [CORE] TG_SHADOW_MODE=1
echo ===============================================================================

:: Set Shadow Mode Flag
set TG_SHADOW_MODE=1
:: Filtrar ?nicamente monedas de alta liquidez para prevenir ruido en el Tensor 10D
set TG_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT

python -O -u main.py --mode futures

echo.
echo ===============================================================================
echo ?? [OMEGA ALARMA] SHADOW ENGINE DETENIDO.
echo 🛑 Bucle infinito desactivado para prevenir cuelgues (Flickering).
echo ===============================================================================
pause
