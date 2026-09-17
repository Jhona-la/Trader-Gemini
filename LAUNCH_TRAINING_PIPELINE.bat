@echo off
echo ========================================================
echo 🧠 DARK ALPHA TRAINING PIPELINE (RUST EDITION)
echo ========================================================
echo.
echo [1/3] Downloading latest historical data...
cargo run --release --bin download_history
if %errorlevel% neq 0 (
    echo [ERROR] download_history failed!
    pause
    exit /b %errorlevel%
)

echo [2/3] Training Dark Alpha Model (Rust Native)...
cargo run --release --bin train_dark_alpha
if %errorlevel% neq 0 (
    echo [ERROR] train_dark_alpha failed!
    pause
    exit /b %errorlevel%
)

echo [3/3] Running Quantum Evolution (Threshold Optimization)...
cargo run --release --bin evolver
if %errorlevel% neq 0 (
    echo [ERROR] evolution_engine failed!
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo ✅ TRAINING PIPELINE COMPLETE! 
echo The AI is now updated. You can run LAUNCH_GOD_MODE.bat
echo ========================================================
pause
