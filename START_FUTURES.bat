@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title [FUTURES] TRADER GEMINI - AUTONOMOUS TRADING (NOUVEAUCRAFT)
color 0B

:LOOP
cls
echo ===============================================================================
echo 🚀 TRADER GEMINI - PROTOCOLO IGNICION-TITAN (FASE 9: METACOGNICION)
echo ===============================================================================
echo Mensaje NouveauCraft: La automatizacion es la madre de la disciplina operativa.
echo.
echo [INFO] System: DYNAMIC ADAPTATION ENABLED + PPO ENTROPY
echo [INFO] Brain:  AUTONOMOUS REGIME DETECTION (XGBoost / RF)
echo [INFO] Mode:   FUTURES (MAINNET/TESTNET DEPENDIENDO DE CONFIG.PY)
echo.
echo [SYSTEM] Activando Entorno Virtual...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] No venv found. Proceeding with global python.
)

echo [SYSTEM] Lanzando Motor Neural HFT (Unbuffered)...
:: Ejecución Optimizada the Python (-O remueve asserts en PROD)
python -O -u main.py --mode futures

echo.
echo ===============================================================================
echo ⚠️ [WARNING] El Motor The Trading ha colapsado o fue Thetenido.
echo 🔄 Auto-Restarting en 3 segundos (Resiliencia Autonoma)...
echo 💡 [TIP] Presiona Ctrl+C repetidamente para asesinar el bucle.
echo ===============================================================================
timeout /t 3
goto LOOP
