@echo off
echo ========================================================
echo 🚀 GOD MODE ACTIVATED - TRADER GEMINI (RUST EDITION)
echo ========================================================
echo.
echo Starting Quantum Engine in Production Mode...

cargo build --release
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile Rust Engine!
    pause
    exit /b %errorlevel%
)

echo [SUCCESS] Engine Compiled. Igniting...
cargo run --release --bin god_engine

pause
