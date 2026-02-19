@echo off
title TRADER GEMINI - SMART GROWTH (PRODUCTION)
color 0B

:LOOP
cls
echo ===============================================================================
echo    TRADER GEMINI - SMART GROWTH PROTOCOL (SAFE OPTIMIZATION)
echo ===============================================================================
echo.
echo [INFO] Starting engine with Optimized Alpha...
echo [INFO] Strategy: SMART GROWTH (Threshold 0.55)
echo [INFO] Network: MAINNET (REAL MONEY)
echo.

python main.py

echo.
echo [WARNING] Bot crashed or stopped. Restarting in 3 seconds...
timeout /t 3
goto LOOP
