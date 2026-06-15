@echo off
title QUANTUM VECTOR TRAINER (FASE 6)
color 0B

echo ==============================================================================
echo  🧬 QUANTUM VECTOR TRAINER - TRADER GEMINI SUPREME 🧬
echo ==============================================================================
echo.
echo Optimizando entorno para maxima velocidad de entrenamiento (XGBoost Hist)...
echo.

set OMP_NUM_THREADS=16
set OPENBLAS_NUM_THREADS=16
set MKL_NUM_THREADS=16
set VECLIB_MAXIMUM_THREADS=16
set NUMEXPR_NUM_THREADS=16

call .venv\Scripts\activate.bat

python scripts\vector_backtest\vector_trainer.py --days 15

echo.
echo Entrenamiento Cuantico Completado.
pause
