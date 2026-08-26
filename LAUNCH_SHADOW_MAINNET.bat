@echo off
color 0E
title TRADER GEMINI V5 - SHADOW MAINNET MODE

echo =======================================================
echo     TRADER GEMINI V5 - 👻 SHADOW MAINNET MODE
echo =======================================================
echo.
echo [1/3] Optimizando arquitectura para CPU Nativa...
set RUSTFLAGS=-C target-cpu=native

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
cargo run --release --bin god_engine
if %errorlevel% neq 0 (
    echo El bot crasheó o se cerró con error. Reiniciando SHADOW en 5 segundos...
    timeout /t 5
    goto SHADOW
)
pause
