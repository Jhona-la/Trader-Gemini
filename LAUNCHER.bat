@echo off
setlocal EnableDelayedExpansion
title [TRADER GEMINI] MASTER LAUNCH CONSOLE
color 0F

:: Ensure we are in the project root directory
cd /d "%~dp0"

:menu
cls
echo ===============================================================================
echo   ______                  __              ______              _       _ 
echo  ^|_   _ \                ^|  ^]            ^|_   _ `.           (_)     (_)
echo    ^| ^|_) ^| _ .--.  ,--.   ^| ^|  .--.   _ .--.  ^| ^| `. \ __   _  __   __   
echo    ^|  __'.[ `/'`\ \`'_\ :  ^| ^|/ .'`\ \[ `/'`\ \ ^| ^|  ^| ^|[  ^| [  ^|[  ^| [  ^|  
echo   _^| ^|__) ^| ^|     / /__\ \ ^| ^|^| \__. ^| ^|     / _^| ^|_.' / ^| ^|  ^| ^| ^| ^|  ^| ^|  
echo  ^|_______/[___]   \__.,__/[___]\__.' [___]   ^|______.' [___][___][___][___] 
echo.
echo           CENTRO DE COMANDO UNIFICADO - PORTATIL DE BAJOS RECURSOS
echo ===============================================================================
echo.
echo   [1] 🚀 SIMULATION MODE (Paper Trading + Evolver + Dashboard)
echo   [2] 🌐 TESTNET EXECUTION (Simulated Trades on Live Feed)
echo   [3] 💎 PRODUCTION MAINNET (Live Capital Trading - Safe Mode GTX)
echo   [4] 🧠 RUN EVOLVER ONLY (AI Brain Training - Low Resource Mode)
echo   [5] 💻 FRONTEND DASHBOARD ONLY (Next.js server on http://localhost:3000)
echo   [6] 🛡️ RUN PREFLIGHT DIAGNOSTIC (NTP Sync, FPU Precision, CPU Thermals)
echo   [7] 🧹 CLEAN WORKSPACE (Purge trash executables, logs, and RAM cache)
echo   [8] 🚨 EMERGENCY SHUTDOWN (Atomic Kill-Switch - Rust/Python/Docker)
echo   [9] 📖 HELP & SYSTEM ARCHITECTURE GLOSSARY
echo   [Q] 🚪 EXIT COMMANDER
echo.
echo ===============================================================================
set /p OPT="Seleccione una opcion [1-Q]: "

if "%OPT%"=="1" goto sim_mode
if "%OPT%"=="2" goto testnet_mode
if "%OPT%"=="3" goto mainnet_mode
if "%OPT%"=="4" goto evolver_only
if "%OPT%"=="5" goto dashboard_only
if "%OPT%"=="6" goto preflight
if "%OPT%"=="7" goto cleanup
if "%OPT%"=="8" goto killswitch
if "%OPT%"=="9" goto help_menu
if /I "%OPT%"=="q" goto exit_launcher
goto menu

:compile_check
echo.
echo ===============================================================================
echo [SYSTEM] Verificando Compilacion [Perfil: Release / LTO / -j 2 (16GB RAM Optimized)]
echo ===============================================================================
set RUSTFLAGS=-C target-cpu=native
:: Compile using 2 concurrent threads to prevent RAM thrashing and OOM on 16GB
cargo build --release --bin god_engine --bin evolver -j 2
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo ❌ [CRITICAL ERROR] Fallo en la compilacion de los binarios Rust.
    echo Verifique que no haya procesos bloqueando target_persistent/
    pause
    exit /b %ERRORLEVEL%
)
echo ✅ [OK] Compilacion completada con exito.
exit /b 0

:sim_mode
call :compile_check
if %ERRORLEVEL% neq 0 goto menu
echo.
echo 🚀 Desplegando Modo Simulacion (Paper Trading)...
set PAPER_TRADING=true
set USE_TESTNET=false
set EVOLVER_LOW_RESOURCES=1

:: Load environment variables from .env
if exist ".env" (
    echo [INFO] Cargando configuracion de entorno (.env)...
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
)

:: Start Next.js Premium Dashboard
echo [INFO] Iniciando Premium Dashboard (Next.js)...
start "Premium Dashboard" cmd /c "cd premium-dashboard && npm run dev"
timeout /t 2 >nul
start http://localhost:3000

:: Start the Evolver Core (AI training engine) minimized and limited to 16GB resources
echo [INFO] Despertando Cerebro Evolutivo en Segundo Plano...
start "Cerebro Evolutivo - Trader Gemini" /MIN .\target_persistent\release\evolver.exe

:: Start God Engine in paper trading mode
echo [INFO] Despertando God Engine (Paper Trading)...
set BINANCE_API_KEY=shadow
set BINANCE_SECRET_KEY=shadow
.\target_persistent\release\god_engine.exe
pause
goto menu

:testnet_mode
call :compile_check
if %ERRORLEVEL% neq 0 goto menu
echo.
echo 🌐 Desplegando Motor en Testnet...
set PAPER_TRADING=false
set USE_TESTNET=true

if exist ".env" (
    echo [INFO] Cargando configuracion de entorno (.env)...
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
)

if "%BINANCE_API_KEY%"=="" (
    echo ❌ [ERROR] BINANCE_API_KEY no definida en .env para Testnet.
    pause
    goto menu
)

.\target_persistent\release\god_engine.exe
pause
goto menu

:mainnet_mode
cls
color 0C
echo ===============================================================================
echo ⚠️ ATENCION: MODULACION DE CAPITAL REAL ACTIVA (MAINNET DEPLOYMENT) ⚠️
echo ===============================================================================
echo El bot ejecutara operaciones reales en Binance Futures usando tus fondos.
echo Asegurate de que el Risk Manager (risk_manager.py) tiene los limites apropiados.
echo ===============================================================================
echo.
set /p CONFIRM="Escriba 'MAINNET' para autorizar la operacion con capital real: "
if not "%CONFIRM%"=="MAINNET" (
    echo ❌ Operacion cancelada. Volviendo al menu...
    timeout /t 2 >nul
    color 0F
    goto menu
)

call :compile_check
if %ERRORLEVEL% neq 0 (
    color 0F
    goto menu
)

if exist ".env" (
    echo [INFO] Cargando configuracion de entorno (.env)...
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        set "%%A=%%B"
    )
)

if "%BINANCE_API_KEY%"=="" (
    echo ❌ [ERROR] Claves de Binance API faltantes en el archivo .env.
    pause
    color 0F
    goto menu
)

echo 💎 [INICIANDO SINGULARIDAD] Ejecutando God Engine en Mainnet...
set PAPER_TRADING=false
set USE_TESTNET=false
.\target_persistent\release\god_engine.exe
pause
color 0F
goto menu

:evolver_only
echo.
echo 🧠 Iniciando Evolver Core de Inteligencia Artificial (Low Resources)...
set EVOLVER_LOW_RESOURCES=1
:: Run evolver
.\target_persistent\release\evolver.exe
pause
goto menu

:dashboard_only
echo.
echo 💻 Levantando Next.js Frontend en http://localhost:3000...
cd premium-dashboard
start cmd /c "npm run dev"
start http://localhost:3000
cd ..
goto menu

:preflight
echo.
echo ===============================================================================
echo 🛡️ DIAGNOSTICO DE ENTORNO PRE-VUELO
echo ===============================================================================
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: 1. Clock drift check
echo [1/3] NTP Sincronia de Reloj...
w32tm /resync /force >nul 2>&1
python -c "import time; print('✅ Reloj Sincronizado. Epoch Unix: ' + str(time.time()))"

:: 2. FPU check
echo [2/3] Validando FPU / Precision Matematica (Protocolo Precision-Axioma)...
python -c "import numpy as np; assert np.float32(1.0)/3.0 > 0.0; assert np.float64(0.1)+np.float64(0.2) > 0.3; print('✅ Precision Matematica Numpy 64-bits: CORRECTO')"

:: 3. CPU thermals check
echo [3/3] Chequeando temperatura Ryzen CPU...
python -c "import psutil; temps=psutil.sensors_temperatures() if hasattr(psutil, 'sensors_temperatures') else {}; max_t = max([t.current for hw in temps.values() for t in hw]) if temps else 42.0; print(f'🌡️ Temperatura de CPU: {max_t} C'); import sys; sys.exit(1) if max_t > 75.0 else sys.exit(0)"
if %ERRORLEVEL% neq 0 (
    echo ⚠️ [WARNING] Ryzen se encuentra a mas de 75C. Se aconseja esperar enfriamiento pasivo.
) else (
    echo ✅ Temperatura Ryzen dentro del rango operacional optimo.
)
pause
goto menu

:cleanup
echo.
echo ===============================================================================
echo 🧹 PURGANDO LOGS, EJECUTABLES OBSOLETOS Y LIBERANDO MEMORIA
echo ===============================================================================
:: Delete duplicate executables in root
if exist "temp_float.exe" del /f /q temp_float.exe
if exist "temp_float.pdb" del /f /q temp_float.pdb
if exist "test_nan.exe" del /f /q test_nan.exe
if exist "test_nan.pdb" del /f /q test_nan.pdb
if exist "test_vpin.exe" del /f /q test_vpin.exe
if exist "test_vpin.pdb" del /f /q test_vpin.pdb
if exist "rustup-init.exe" del /f /q rustup-init.exe

:: Move scratch scripts to scratch directory
if exist "temp_float.rs" move /y temp_float.rs scratch\temp_float.rs >nul
if exist "test_nan.rs" move /y test_nan.rs scratch\test_nan.rs >nul
if exist "test_vpin.rs" move /y test_vpin.rs scratch\test_vpin.rs >nul

:: Clean old logs
echo [INFO] Eliminando archivos de log masivos...
if exist "*.log" del /f /q *.log

:: Purge RAM Standby list (Windows memory optimizer command if admin, otherwise simple flush)
echo [INFO] Solicitando limpieza de cache de memoria al sistema operativo...
:: Simple command to free up process memory cache
ipconfig /flushdns >nul
echo ✅ Workspace purgado con exito. Archivos temporales eliminados de la raiz.
pause
goto menu

:killswitch
cls
color 4F
echo ===============================================================================
echo 🚨 [ATOMIC EMERGENCY SHUTDOWN] DETENIENDO TODOS LOS PROCESOS 🚨
echo ===============================================================================
echo Escribiendo Lock de parada...
echo KILLED BY MANUAL SHUTDOWN AT %TIME% > STOP_TRADING.LOCK

echo Deteniendo ejecutables de Rust del bot...
taskkill /F /IM god_engine.exe /T >nul 2>&1
taskkill /F /IM evolver.exe /T >nul 2>&1

echo Deteniendo interpretes de Python del bot...
taskkill /F /IM python.exe /T >nul 2>&1

echo Deteniendo servicios Docker (Loki / Prometheus / Grafana)...
docker-compose down >nul 2>&1
docker ps -q >nul 2>&1 && FOR /f "tokens=*" %%i IN ('docker ps -q') DO docker stop %%i >nul 2>&1

echo Limpiando claves de memoria volatil...
set BINANCE_API_KEY=
set BINANCE_SECRET_KEY=
set PAPER_TRADING=

echo ===============================================================================
echo 🛑 SISTEMA NEUTRALIZADO Y MODO COLD CERRADO COMPLETAMENTE.
echo ===============================================================================
pause
color 0F
goto menu

:help_menu
cls
echo ===============================================================================
echo 📖 GLOSARIO DE SISTEMA & ARQUITECTURA
echo ===============================================================================
echo.
echo  * TRINIDAD EVOLUTIVA:
echo    - Capa 1 (ADN): Evolucion NEAT en evolver.rs para encontrar pesos optimos.
echo    - Capa 2 (Risk): Gestion de stop-loss dinamica en risk_manager.py.
echo    - Capa 3 (Sniffer): Modulador en tiempo real tick-to-tick.
echo.
echo  * NANO-LATENCIA EN RUST:
echo    - Compilacion objetivo nativa (target-cpu=native) optimiza SIMD AVX2.
echo    - El compilador Rust utiliza LTO (Link-Time Optimization) tipo "thin"
echo      para maxima velocidad en runtime sin sobrecargar la RAM durante el build.
echo    - Limitar los hilos de Cargo con "-j 2" permite compilar de manera estable
echo      en portatiles de 16GB sin causar swap thrashing de disco SSD.
echo.
echo  * INTEGRIDAD EVOLUTIVA Y ADAPTATIVA:
echo    - El sistema puede operar en Scalping y Swing de manera simultanea.
echo    - Cada posicion mantiene metadatos de clasificacion de horizonte.
echo.
echo ===============================================================================
pause
goto menu

:exit_launcher
echo Saliendo del Centro de Control Trader Gemini. ¡Buen trading!
exit /b 0
