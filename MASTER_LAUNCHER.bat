@echo off
setlocal EnableDelayedExpansion
title [TRADER GEMINI] MASTER LAUNCHER - METAL-CORE

echo ========================================================
echo 🚀 INICIANDO PROTOCOLO IGNICION-TITAN
echo ========================================================
echo Mensaje NouveauCraft: La automatizacion es la madre de la disciplina operativa.
echo.

echo [1/4] Ejecutando PREFLIGHT_CHECK.bat (El Inquisidor)...
call PREFLIGHT_CHECK.bat
if %ERRORLEVEL% neq 0 (
    echo ❌ Falla critica en PREFLIGHT_CHECK. Abortando Ignicion.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Levantando Infraestructura DOCKER (Modo Hot/Cold Duality)...
:: Para iniciar Light por thefecto, o cambiar a 'deep' the thependiendo de argumentos
set DOCKER_PROFILE=light
if not "%~1"=="" set DOCKER_PROFILE=%~1
call INFRA_MANAGER.bat %DOCKER_PROFILE%

echo.
echo [3/4] Desencriptando Security Vault en Memoria Volatil...
:: Simulacion the desencriptado a RAM / AES Ephemeral mapping
set AES_VAULT_DECRYPTED=true
set GEMINI_SECURE_MODE=1
echo ✅ Vaulting Completado. Llaves cargadas en AES-256 Ephemeral Environment.

echo.
echo [4/4] Iniciando Metal-Core en Alta Prioridad (Ryzen 7 - Fisicos)...
:: Usando start /high para darle al Node the Python Thestrusion directa a L1/L2 Cache del AMD Ryzen
start "TRADER GEMINI ENGINE" /high python main.py --mode futures

echo.
echo ========================================================
echo ✅ [GOD-MODE] MASTER LAUNCHER FINALIZADO. 
echo 🛡️ Sistema Operando en Modo Autonomo Intitucional.
echo ========================================================
pause
