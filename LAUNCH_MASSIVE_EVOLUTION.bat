@echo off
title GOD MODE - MASS HYPER EVOLVER
color 0D

echo ===============================================================================
echo  OMNISCIENCIA 30D: MASS HYPER EVOLUTION (OPTUNA)
echo ===============================================================================
echo.
echo Iniciando el simulador cuantico masivo para optimizar el Top 10 de monedas.
echo ATENCION: Ejecutando 100 iteraciones por moneda.
echo Esto ejecutara simulaciones altamente optimizadas. Puedes dejarlo corriendo.
echo.

call .venv\Scripts\activate.bat
python scripts\mass_hyper_evolver.py --days 15 --trials 100

echo.
echo ===============================================================================
echo  EVOLUCION COMPLETADA. REVISA config/genotypes/
echo ===============================================================================
pause
