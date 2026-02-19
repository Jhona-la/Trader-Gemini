@echo off
setlocal
cd /d "%~dp0"
title [SPOT] TRADER GEMINI - DYNAMIC EVOLUTION
color 0A

:LOOP
cls
echo ===============================================================================
echo    TRADER GEMINI - DYNAMIC EVOLUTION (SPOT MODE)
echo ===============================================================================
echo.
echo [INFO] System: DYNAMIC ADAPTATION ENABLED
echo [INFO] Brain:  AUTONOMOUS REGIME DETECTION
echo [INFO] Mode:   SPOT (MAINNET)
echo.

call .venv\Scripts\activate.bat

python -O -u main.py --mode spot

echo.
echo [WARNING] Engine stopped. Auto-Restarting in 3 seconds...
timeout /t 3
goto LOOP
