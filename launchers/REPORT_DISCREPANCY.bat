@echo off
setlocal EnableDelayedExpansion
title [BLOCK G] REPORTE DE DISCREPANCIA (TESTNET VS BACKTEST)
color 0B

echo ===============================================================================
echo 📊 PROTOCOLO BLOCK G: DISCREPANCY REPORT GENERATOR (THE ORACLE)
echo ===============================================================================
echo Mensaje NouveauCraft: Un ratio de Sharpe no auditado es simple fantasia.
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo ❌ [FATAL] Environment virtual no thetectado. Lanzamiento Abortado.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

echo 🔍 Consolidando datos the PnL the 72 Horas (Live Status)...
echo 🔍 Cruzando resultados contra Historical Backtest JSON...
echo.

python scripts/generate_discrepancy_report.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Falla en la generacion thel reporte the discrepancia.
) else (
    echo.
    echo ✅ El thescrito the Discrepancia ha sido forjado exitosamente thentro the la carpeta analysis/
)

echo ===============================================================================
pause

