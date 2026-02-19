@echo off
setlocal
cd /d "%~dp0"
title [GOD MODE] TRADER GEMINI - DYNAMIC EVOLUTION (OMEGA)
color 0E

echo ========================================================
echo    TRADER GEMINI - OMEGA PROTOCOL (PHASE 10)
echo ========================================================
echo.

:: 1. Environment Check
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual Environment not found! Launch aborted.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

:: 2. Trinity Model Verification (Pre-Flight) - RUN ONCE
echo [SYSTEM] Verifying Evolution Trinity Models...
if not exist ".models" (
    echo [WARNING] Brain models not found. Evolution Engine will start from Genesis ADN.
) else (
    echo [OK] DNA Genotypes found.
)

:: 3. God-Mode Pre-Flight Audit - RUN ONCE
echo.
echo [SYSTEM] Running God-Mode Micro-Latency Audit...
python core/pre_flight.py
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo [FATAL] GOD MODE AUDIT FAILED! 
    pause
    exit /b 1
)

:LOOP
:: 4. Launch Loop
cls
echo ===============================================================================
echo    TRADER GEMINI - DYNAMIC EVOLUTION ENGINE (GOD MODE ACTIVE)
echo ===============================================================================
echo.
echo [INFO] System: DYNAMIC ADAPTATION ENABLED
echo [INFO] Mode:   GOD MODE (HIGH PRIORITY + OPTIMIZED)
echo.
echo [CORE] CPU Affinity: Automatic Pinning
echo [CORE] Bytecode: Optimized (-O)
echo [CORE] Buffer: Unbuffered (-u)
echo.

:: Launch directly (no 'start') to capture exit code and allow loop
python -O -u main.py --mode futures

echo.
echo [WARNING] Omega Engine stopped. Auto-Restarting in 3 seconds...
timeout /t 3
goto LOOP
