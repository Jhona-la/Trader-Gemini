@echo off
echo ========================================
echo  TRADER GEMINI - SPOT INSTANCE
echo ========================================
echo Starting Spot Trading Bot...
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
echo 🟡 STARTING TRADER GEMINI - SPOT MODE
python main.py --mode spot
pause
