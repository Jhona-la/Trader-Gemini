@echo off
color 0E
title TRADER GEMINI V5 - SHADOW MAINNET MODE

echo =======================================================
echo     TRADER GEMINI V5 - 👻 SHADOW MAINNET MODE
echo =======================================================
echo.
echo [1/3] Compilando en el target\ compartido (CPU nativa via .cargo\config.toml)...
:: RB-H1 (2026-10-19): SIN RUSTFLAGS de entorno. La variable reemplazaba los
:: rustflags de .cargo/config.toml y obligaba a recompilar todo el arbol con
:: una huella distinta a la de los agentes.
cargo build --release --bin god_engine -j 2
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la compilacion de god_engine.
    pause
    exit /b 1
)

echo [2/3] Levantando Servidor de Telemetria en Background...
start "" "http://127.0.0.1:3000"

:SHADOW
echo.
echo =======================================================
echo  👻 ATENCION: Entrando a FASE SHADOW MAINNET.
echo  Se procesaran datos de MAINNET, pero las ordenes 
echo  seran interceptadas (VIRTUALIZADAS) sin arriesgar capital.
echo =======================================================
set SHADOW_MODE=true
set USE_TESTNET=false
set BINANCE_USE_DEMO=false
:: RB-H1: se ejecuta una COPIA (despliegue\) para no bloquear el .exe del
:: target\ compartido mientras el motor corre (OS Error 32 en los builds).
if not exist "despliegue" mkdir "despliegue"
copy /Y "target\release\god_engine.exe" "despliegue\god_engine.exe" >nul
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo copiar a despliegue\ ^(hay un motor vivo?^). Cierralo y reintenta.
    pause
    exit /b 1
)
.\despliegue\god_engine.exe
if %errorlevel% neq 0 (
    echo El bot crasheó o se cerró con error. Reiniciando SHADOW en 5 segundos...
    timeout /t 5
    goto SHADOW
)
pause
