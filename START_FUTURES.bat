@echo off
setlocal
cd /d "%~dp0"
title [FUTURES] TRADER GEMINI - DYNAMIC EVOLUTION
color 0B

:LOOP
cls
echo ===============================================================================
echo    TRADER GEMINI - DYNAMIC EVOLUTIONARY ADAPTATION (PHASE 10)
echo ===============================================================================
echo.
echo [INFO] System: DYNAMIC ADAPTATION ENABLED
echo [INFO] Brain:  AUTONOMOUS REGIME DETECTION
echo [INFO] Mode:   FUTURES (MAINNET)
echo.
echo [SYSTEM] Launching Neural Engine...

call .venv\Scripts\activate.bat

:: Launch with High Priority, Optimized, Unbuffered
python -O -u main.py --mode futures

echo.
echo [WARNING] Engine stopped or crashed. Auto-Restarting in 3 seconds...
echo [TIP] Press Ctrl+C to terminate the loop.
timeout /t 3
goto LOOP
