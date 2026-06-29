@echo off
setlocal EnableDelayedExpansion
echo =======================================================
echo     TRADER GEMINI - GOD ENGINE [PRODUCTION MAINNET]
echo =======================================================
echo.

if not exist ".env" (
    echo [!] ERROR: No se encontro el archivo .env
    echo Creando archivo .env desde .env.template...
    copy .env.template .env
    echo [!] Por favor edita el archivo .env con tus llaves reales de Binance Futures.
    echo Presiona cualquier tecla para salir...
    pause >nul
    exit /b 1
)

echo [1] Cargando llaves de Produccion desde .env...
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
    set "%%A=%%B"
)

if "%BINANCE_API_KEY%"=="YOUR_PRODUCTION_API_KEY_HERE" (
    echo [!] ERROR: Debes editar el .env y colocar tus llaves reales.
    pause >nul
    exit /b 1
)

echo [2] Aislado el entorno de Compilacion (Bypass OS Error 32)...
set CARGO_TARGET_DIR=target_prod

echo [3] Forjando binario Quantico (Release Mode)...
cargo build --release --bin god_engine

if %ERRORLEVEL% neq 0 (
    echo [!] ERROR: Fallo en la compilacion de Rust.
    pause >nul
    exit /b %ERRORLEVEL%
)

echo.
echo [4] INICIANDO SINGULARIDAD (DINERO REAL ACTIVADO)
echo PRESIONA CTRL+C PARA ABORTAR INMEDIATAMENTE
timeout /t 5

.\target_prod\release\god_engine.exe
pause
