@echo off
setlocal
echo =========================================================
echo TRADER GEMINI: QUANTUM VECTOR-JIT BACKTEST (FASE 5)
echo =========================================================
echo.
echo Motor de Nano-Latencia Inicializando...
echo.

call .venv\Scripts\activate

set OMP_NUM_THREADS=16
set MKL_NUM_THREADS=16
set OPENBLAS_NUM_THREADS=16
set VECLIB_MAXIMUM_THREADS=16
set NUMEXPR_NUM_THREADS=16
set NUMBA_NUM_THREADS=16

python scripts\fast_vector_backtest.py

echo.
echo Simulacion Cuantica Completada.
pause
