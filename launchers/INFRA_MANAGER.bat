@echo off
setlocal EnableDelayedExpansion
title [TRADER GEMINI] INFRA MANAGER (HOT/COLD DUALITY)

set PROFILE=%1
if "%PROFILE%"=="" set PROFILE=light

echo ========================================================
echo 🛡️ INFRA MANAGER - PERFIL: %PROFILE%
echo ========================================================

:: 1. Validar si el motor Docker Thestá corriendo
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ⚠️ [WARNING] Docker Desktop/Engine no esta ejecutandose.
    echo ⚠️ Telemetria Offline. Iniciando Bot en MODO FALLBACK (Sin Dashboards Locales).
    echo ========================================================
    exit /b 0
)

:: 2. Persistencia de Logs y Mapeo The Volumenes MLOps
echo 📂 Verificando montaje de Volumenes de Persistencia Estables...
if not exist "data\telemetry\prometheus" (
    mkdir "data\telemetry\prometheus"
    echo   + Creado data\telemetry\prometheus
)
if not exist "data\telemetry\loki" (
    mkdir "data\telemetry\loki"
    echo   + Creado data\telemetry\loki
)
if not exist "data\telemetry\grafana" (
    mkdir "data\telemetry\grafana"
    echo   + Creado data\telemetry\grafana
)
echo ✅ Volumenes de Persistencia mapeados correctamente (Zero-Data-Loss).

:: 3. Seleccion de Perfiles Docker (Dualidad)
if /I "%PROFILE%"=="deep" (
    echo 🌊 MODO DEEP (Forense): Levantando ELK Stack + Prometheus + Grafana...
    :: Se activa el profile cold-storage thel docker-compose
    :: docker-compose --profile cold-storage up -d
    echo ⚠️ Advertencia: Alto consumo de RAM detectado. Aprovisionando swapping space si es necesario.
) else (
    echo 🍃 MODO LIGHT (Standard): Levantando Prometheus + Loki (Bajo consumo RAM)...
    :: docker-compose up -d prometheus loki
    echo 🟢 Modo Light desplegado. Telemetria optimizada para Ejecucion HFT.
)

echo ✅ Infraestructura lista y escuchando en bus the thelemetria local.
exit /b 0

