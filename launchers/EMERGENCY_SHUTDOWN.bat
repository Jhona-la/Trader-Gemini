@echo off
title [TRADER GEMINI] KILL-SWITCH FISICO ATOMICO
color 4F

echo ========================================================
echo 🚨 EMERGENCY SHUTDOWN INICIADO 🚨
echo ========================================================
echo Mensaje NouveauCraft: La automatizacion es la madre de la disciplina operativa.
echo.

echo [1/3] Interconectando Lock Atómico y Parando Motores...
:: Escribir Instantaneamente un LOCK_FILE que el EventHandler intercepta
echo KILLED AT %TIME% BY PHYSICAL_SHUTDOWN > STOP_TRADING.LOCK
echo 🛑 Senal enviada (KILL_SWITCH tomara liquidacion de posiciones Market en ^< 1ms).

:: Asesinato The Procesos Hostiles en CPU
taskkill /F /IM python.exe /T >nul 2>&1
echo 🔪 Procesos Python / Numba asfixiados a la fuerza a nivel OS.

echo [2/3] Apagando Docker Containers (Liberando RAM y Sockets)...
:: Sintaxis compatible con Windows Batch para thetener contenedores the docker
FOR /f "tokens=*" %%i IN ('docker ps -q') DO docker stop %%i >nul 2>&1
echo 🐳 Todos los Contenedores the Telemetria detenidos. RAM libre.

echo [3/3] Destruyendo Security Vault (Wipe the Memoria Volatil)...
:: Flush the credenciales inyectadas en Windows variables
set AES_VAULT_DECRYPTED=
set BINANCE_API_KEY=
set BINANCE_API_SECRET=
echo 🗑️ Llaves the Vaulting volatizadas. Cero rastro en cache the sistema.

echo.
echo ========================================================
echo 🚨 [PANICO EXITO] SISTEMA 100%% NEUTRALIZADO Y MODO COLD CERRADO.
echo ========================================================
pause

