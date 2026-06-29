@echo off
echo =======================================================
echo [TRADER GEMINI V5] FORJANDO EL BINARIO MAESTRO (LLVM)
echo =======================================================

echo.
echo [1/3] Limpiando artefactos previos...
cargo clean

echo.
echo [2/3] Configurando directivas extremas de LLVM...
:: Forzamos al compilador a usar las instrucciones vectoriales especificas de esta CPU (AVX, SSE)
set RUSTFLAGS=-C target-cpu=native

echo.
echo [3/3] Compilando con LTO Fat y Opt-Level 3...
cargo build --release

if %ERRORLEVEL% GEQ 1 (
    echo.
    echo [ERROR] La compilacion ha fallado. Revisa la salida de Rust.
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [TRADER GEMINI V5] COMPILACION EXITOSA
echo Binario ubicado en: target\release\trader-gemini-v5.exe
echo =======================================================
pause
