@echo off
setlocal
color 0A
title Trader Gemini - Quantum Vector Trainer

echo =======================================================
echo 🧬 INICIANDO ENTRENAMIENTO VECTORIAL CUANTICO (Fase 6)
echo =======================================================
echo.
echo Esto entrenara los modelos XGBoost de alta velocidad 
echo en formato binario compatible con JIT.
echo.

set PYTHONPATH=%cd%
call .venv\Scripts\activate

python scripts\vector_backtest\vector_trainer.py --days 30

echo.
echo =======================================================
echo ✅ Entrenamiento Terminado. Los pesos estan en .models/
echo =======================================================
pause
