@echo off
echo ===============================================================================
echo  OMNISCIENCIA 30D: PARALLEL HYPER EVOLUTION (OPTUNA CLUSTER)
echo ===============================================================================
echo.
echo Iniciando cluster cuantico de 4 nodos para aceleracion masiva.
echo Esto ejecutara multiples procesos simultaneamente.
echo.

call .venv\Scripts\activate.bat

:: Launch 4 parallel processes in the background sharing the same SQLite DB
start /b python scripts\mass_hyper_evolver.py --days 15 --trials 100
start /b python scripts\mass_hyper_evolver.py --days 15 --trials 100
start /b python scripts\mass_hyper_evolver.py --days 15 --trials 100
start /b python scripts\mass_hyper_evolver.py --days 15 --trials 100

echo.
echo Cluster desplegado con exito. Los procesos estan optimizando en segundo plano.
echo Revisa la salida de la consola o la base de datos Optuna para el progreso.
echo ===============================================================================
