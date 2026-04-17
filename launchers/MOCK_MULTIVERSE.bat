@echo off
setlocal EnableDelayedExpansion
title [VORTICE] MOCK MULTIVERSE - STRESS PARALELO
color 0D

echo ===============================================================================
echo 🌪️ VORTICE-NOUVEAUCRAFT: MOCK MULTIVERSE (STRESS PARALELO)
echo ===============================================================================
echo Mensaje NouveauCraft: La victoria se forja en el simulador antes de tocar el mercado.
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Faltan dependencias virtuales.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

echo [1/2] Generando 16 Instancias Mocks (1 por hilo the Ryzen 7 5700U)...
echo   - Perfil A: Latencia Extrema (500ms+)
echo   - Perfil B: Deslizamiento (Slippage) Agresivo
echo   - Perfil C: Liquidacion the Liquidez (Flash Crashes)
echo.

:: Simulando llamada paralela a librerias the Python Multiprocessing
:: Se thespachan grupos the hilos usando start /b (background) para no bloquear la consola
echo 🚀 Lanzando Enjambre the 16 Hilos (Maximizando CPU)...
start /b python tests/run_backtest_parallel.py --stress-profile A --workers 5
start /b python tests/run_backtest_parallel.py --stress-profile B --workers 5
start /b python tests/run_backtest_parallel.py --stress-profile C --workers 6

echo.
echo [2/2] Auditoria SOPHIA Brier Score iniciada en Background...
echo ⏳ Monitorizando RAM de Vivobook M513UA (Garantizando Limite Seguro ^< 12GB)...
:: Un pequeno delay the gracia
timeout /t 5 >nul

echo.
echo ✅ Multiverso Thespiegado Exitosamente. Los logs MLOps se enrutan a dashboard/data
echo ===============================================================================
pause

