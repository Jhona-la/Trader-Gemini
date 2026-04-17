@echo off
setlocal
title [HELP] TRADER GEMINI - INSTITUTIONAL COMANDOS
color 0F

:menu
cls
echo ========================================================
echo    TRADER GEMINI - ÍNDICE DE COMANDOS (OMEGA)
echo ========================================================
echo.
echo  1. [INFO]   Qué es Metal-Core y la Trinidad?
echo  2. [INSTAL] Requisitos y Setup Inicial
echo  3. [LAUNCH] Lanzamiento de Producción (God Mode)
echo  4. [AUDIT]  Comandos de Certificación y Tests
echo  5. [TERM]   Glosario de Nano-Latencia HFT
echo  Q. [SALIR]  Regresar a la terminal
echo.
set /p opt="Seleccione una opción: "

if "%opt%"=="1" goto info
if "%opt%"=="2" goto instal
if "%opt%"=="3" goto launch
if "%opt%"=="4" goto audit
if "%opt%"=="5" goto term
if "%opt%"=="q" exit /b
if "%opt%"=="Q" exit /b
goto menu

:info
cls
echo [TRINIDAD EVOLUTIVA]
echo Capa 1: ADN Genético (Optimización de parámetros)
echo Capa 2: Refuerzo (RL) (Control táctico de stops)
echo Capa 3: Online Learning (Ajuste tick-to-tick)
echo.
echo [METAL-CORE]
echo Arquitectura Zero-Copy, Numba JIT y Ring Buffers.
echo Latencia certificada: 2.3 microsegundos.
pause
goto menu

:instal
cls
echo [INSTALACIÓN]
echo 1. python -m venv .venv
echo 2. .\.venv\Scripts\activate.ps1
echo 3. pip install -r requirements.txt
echo 4. Configura .env con tus credenciales.
pause
goto menu

:launch
cls
echo [LANZAMIENTO]
echo [GOD MODE] -> LAUNCH_GOD_MODE.bat (Recomendado)
echo [FUTURES]  -> START_FUTURES.bat
echo [MANUAL]   -> python -O main.py --mode futures
pause
goto menu

:audit
cls
echo [AUDITORÍA Y TEST]
echo 💎 Certificación Final: tests/certification_of_perfection.py
echo ⚡ Test de Latencia: tests/audit_system_latency.py
echo 🛡️ Check de Salud: utils/health_check.py
pause
goto menu

:term
cls
echo [GLOSARIO HFT]
echo - Jitter: Inestabilidad en la latencia de red/proceso.
echo - Zero-Copy: Evitar copia de datos para ahorrar nanosegundos.
echo - Ring Buffer: Cola circular de tamaño fijo ultra-rápida.
echo - Hot-Path: El código crítico que se ejecuta cada milisegundo.
pause
goto menu

