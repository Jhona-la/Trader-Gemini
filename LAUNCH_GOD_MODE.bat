@echo off
title GOD ENGINE L3 - DEMO por defecto (mainnet requiere --force-live + MAINNET_ARMED)
color 0A
cls

REM ============================================================
REM HOST-011 / HOST-012 (INFORME DECIMOCUARTO) — LEER ANTES:
REM
REM [HOST-011] ESTE LAUNCHER OPERA EN **DEMO/TESTNET POR DEFECTO**.
REM            El titulo anterior ("Mainnet Execution / REAL FUNDS
REM            ACTIVE") era DESINFORMACION: este .bat NO pasa
REM            --force-live. Para mainnet real se necesita:
REM              1) --force-live en la linea de comandos, Y
REM              2) el archivo fisico config_dir\MAINNET_ARMED
REM            Sin ambos, el motor cae a DEMO (Plan Maestro 7.4).
REM
REM [HOST-012] NO ejecutar este launcher desde run_god_engine.ps1:
REM            ese script INYECTA CLAVES "shadow" (BINANCE_API_KEY=shadow)
REM            que TIENEN PRIORIDAD SOBRE .env (dotenvy no sobreescribe
REM            variables ya presentes). NUNCA USAR PARA MAINNET.
REM            Para produccion: usar LAUNCH_PRODUCTION.bat SIN ese script.
REM ============================================================

echo ========================================================
echo                 T R A D E R   G E M I N I
echo                  L A U N C H   M A S T E R
echo ========================================================
echo.
echo  [!] ADVERTENCIA HOST-011: modo DEMO/TESTNET por defecto.
echo      Este launcher NO pasa --force-live. Mainnet real exige
echo      --force-live + config_dir\MAINNET_ARMED (armado humano).
echo.
echo [1] Compilando Binario Rust [Perfil: Ultra-Release / LTO / Abort]
echo.

:: RB-H1 (2026-10-19): UN SOLO target\ compartido (antes CARGO_TARGET_DIR=
:: target_release_god duplicaba el arbol release completo, >8 GB). -j 2 por
:: el presupuesto de 16 GB de RAM (AGENT_SYNC §4).
cargo build --release --bin god_engine -j 2
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [CRITICAL ERROR] Fallo en la compuerta de compilacion cuantica.
    pause
    exit /b %ERRORLEVEL%
)
:: Se ejecuta una COPIA desde despliegue\: el .exe de target\release nunca
:: queda bloqueado (OS Error 32). Si ya hay un motor vivo la copia falla.
if not exist "despliegue" mkdir "despliegue"
copy /Y "target\release\god_engine.exe" "despliegue\god_engine.exe" >nul
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [CRITICAL ERROR] No se pudo copiar a despliegue\ ^(hay un motor vivo?^). Cierralo y reintenta.
    pause
    exit /b 1
)

echo.
echo [2] Compilacion Exitosa. Desplegando Dashboard L3 (TUI Matrix)...
echo.

:: Abre el dashboard en el navegador por defecto (es asincrono)
start http://localhost:8080

echo.
echo [3] Iniciando IA Engine y Motores de Red (DEMO/TESTNET por defecto).
echo ========================================================
echo   NOTA: sin --force-live + config_dir\MAINNET_ARMED esto NO
echo   opera fondos reales, aunque el banner anterior dijera lo
echo   contrario (HOST-011).
echo ========================================================
echo.

:: Ejecuta el bot. En caso de crash el terminal quedara abierto.
.\despliegue\god_engine.exe

if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [CRITICAL PANIC] Engine Exited Abnormally! Code: %ERRORLEVEL%
)

echo.
pause
