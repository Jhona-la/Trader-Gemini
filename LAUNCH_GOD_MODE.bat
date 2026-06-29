@echo off
title GOD ENGINE L3 - Mainnet Execution
color 0A
cls

echo ========================================================
echo                 T R A D E R   G E M I N I 
echo                  L A U N C H   M A S T E R 
echo ========================================================
echo.
echo [1] Compilando Binario Rust [Perfil: Ultra-Release / LTO / Abort]
echo.

set CARGO_TARGET_DIR=target_release_god
cargo build --release --bin god_engine -j 1
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [CRITICAL ERROR] Fallo en la compuerta de compilacion cuantica.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2] Compilacion Exitosa. Desplegando Dashboard L3 (TUI Matrix)...
echo.

:: Abre el dashboard en el navegador por defecto (es asincrono)
start http://localhost:8080

echo.
echo [3] Iniciando IA Engine y Motores de Red en Mainnet.
echo ========================================================
echo                  WARNING: REAL FUNDS ACTIVE
echo ========================================================
echo.

:: Cargar variables de entorno desde .env
if exist ".env" (
    echo [INFO] Cargando configuracion de entorno (.env)...
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        :: Evitar comentarios
        echo %%A | findstr /b /c:"#" >nul || (
            set "%%A=%%B"
        )
    )
) else (
    echo [WARNING] No se encontro archivo .env
)

:: Ejecuta el bot. En caso de crash el terminal quedara abierto.
.\target_release_god\release\god_engine.exe

if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [CRITICAL PANIC] Engine Exited Abnormally! Code: %ERRORLEVEL%
)

echo.
pause
