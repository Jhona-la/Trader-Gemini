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

:: RB-H1 (2026-10-19): el OS Error 32 (exe bloqueado por la instancia viva)
:: se evita ejecutando una COPIA desde despliegue\, no con un target\ aparte
:: (CARGO_TARGET_DIR=target_prod duplicaba el arbol release completo, >8 GB).
echo [2] Un solo target\ compartido; se ejecuta la copia de despliegue\...

echo [3] Forjando binario Quantico (Release Mode)...
cargo build --release --bin god_engine -j 2

if %ERRORLEVEL% neq 0 (
    echo [!] ERROR: Fallo en la compilacion de Rust.
    pause >nul
    exit /b %ERRORLEVEL%
)
if not exist "despliegue" mkdir "despliegue"
copy /Y "target\release\god_engine.exe" "despliegue\god_engine.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo [!] ERROR: No se pudo copiar a despliegue\ ^(hay un motor vivo?^). Cierralo y reintenta.
    pause >nul
    exit /b 1
)

echo.
echo [4] INICIANDO SINGULARIDAD (DINERO REAL ACTIVADO)
echo PRESIONA CTRL+C PARA ABORTAR INMEDIATAMENTE
timeout /t 5

.\despliegue\god_engine.exe
pause
