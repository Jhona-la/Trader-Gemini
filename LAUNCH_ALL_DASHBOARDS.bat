@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title [DASHBOARDS] TRADER GEMINI
color 0B

echo ===============================================================================
echo 🚀 INICIANDO ENTORNO DE MONITOREO OMEGA
echo ===============================================================================

:: 1. Iniciar Next.js Premium Web App
echo [1/3] Lanzando Next.js Premium Dashboard...
cd premium-dashboard
start "Trader Gemini Premium" cmd /c "npm run dev -- -p 3001"
cd ..

:: 2. Iniciar Prometheus (si existe)
echo [2/3] Lanzando Telemetria Prometheus...
if exist "monitoring_tools\prometheus-2.53.0.windows-amd64\prometheus.exe" (
    cd monitoring_tools\prometheus-2.53.0.windows-amd64
    start "Prometheus Telemetry" cmd /c "prometheus.exe --config.file=prometheus.yml"
    cd ..\..
) else (
    echo [!] Prometheus no encontrado en la ruta esperada.
)

:: 3. Iniciar Grafana (si existe)
echo [3/3] Lanzando Grafana Server...
if exist "monitoring_tools\grafana-v11.1.0\bin\grafana-server.exe" (
    cd monitoring_tools\grafana-v11.1.0\bin
    start "Grafana Server" cmd /c "grafana-server.exe"
    cd ..\..\..
) else (
    echo [!] Grafana no encontrado en la ruta esperada.
)

echo.
echo ✅ Todos los dashboards han sido iniciados.
echo.
echo 🌐 Accesos:
echo - Premium Web App: http://localhost:3001
echo - Grafana Server:  http://localhost:3000
echo.
pause
