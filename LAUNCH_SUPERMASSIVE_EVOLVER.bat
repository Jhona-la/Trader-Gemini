@echo off
echo ======================================================================
echo 🌌 OMNI-EVOLVER SUPERMASSIVE UNIVERSAL RUNNER 🌌
echo ======================================================================
echo.
echo ⚠️  ADVERTENCIA: Esta simulación es SUPERMASIVA.
echo Se evaluaran 6 monedas durante 15 dias historicos en 300 generaciones.
echo El proceso tomara multiples horas. La memoria RAM se mantendra segura.
echo.
echo Presiona cualquier tecla para iniciar el Big Bang Computacional...
pause >nul

echo Activando entorno virtual si existe...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Lanzando Simulacion Cuantica...
python scripts\global_omni_evolver.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT,XRPUSDT --days 15 --trials 300

echo.
echo ======================================================================
echo 🏆 SIMULACION SUPERMASIVA COMPLETADA 🏆
echo Revisa la carpeta data/ para encontrar tu Genoma Dorado Universal.
echo ======================================================================
pause
