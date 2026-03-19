@echo off
setlocal
title [UMBRAL-SINCRO] AUDITORIA DE ENTORNO Y OPERADOR

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python scripts/umbral_sincro.py
if %ERRORLEVEL% neq 0 (
    echo [FATAL] Falla en Protocolo UMBRAL-SINCRO. Abortando.
)
pause
