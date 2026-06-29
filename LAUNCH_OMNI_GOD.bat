@echo off
setlocal EnableDelayedExpansion
color 0B
title 🚀 TRADER GEMINI - QUANTUM CORE (Fase 19) 🚀

echo ========================================================
echo        _________    ____  ____  ____  ___
echo       / ____/   ^|  / __ \/ __ \/ __ \/ __ \
echo      / / __/ /^| ^| / /_/ / /_/ / / / / / / /
echo     / /_/ / ___ ^|/ _, _/ ____/ /_/ / /_/ / 
echo     \____/_/  ^|_/_/ ^|_^|_/    \____/\____/  
echo.
echo           QUANTUM GOD ENGINE - TESTNET
echo ========================================================
echo.

if not exist ".env" (
    echo [ERROR] No se encontro el archivo .env.
    echo Por favor crea un archivo .env con tus claves:
    echo BINANCE_API_KEY=tu_api_key_testnet
    echo BINANCE_API_SECRET=tu_api_secret_testnet
    echo.
    pause
    exit /b
)

echo [1/3] Verificando compilacion de Maximo Rendimiento (Release)...
cargo build --release --bin god_engine
if %ERRORLEVEL% neq 0 (
    echo [ERROR] La compilacion ha fallado. Revisa los logs.
    echo TIP: Asegurate de no tener ningun IDE que bloquee el directorio target/.
    pause
    exit /b
)
echo [OK] Compilacion completada.
echo.

echo [2/3] Iniciando la Matriz del Dashboard...
echo Se abrira tu navegador en breve...
timeout /t 2 >nul
start http://localhost:8080

echo [3/3] Despertando el GOD ENGINE (Testnet Execution)...
echo ========================================================
echo ADVERTENCIA: La ejecucion iniciara en modo Testnet usando la 
echo red WS de Produccion para lecturas O(1).
echo ========================================================
echo.

:: Forzamos el target dir epimero por si el entorno no cargo config.toml
set CARGO_TARGET_DIR=target_persistent
cargo run --release --bin god_engine

pause
