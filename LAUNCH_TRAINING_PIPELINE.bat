@echo off
echo ========================================================
echo 🧠 NANO-FOREST TRAINING PIPELINE (RUST EDITION)
echo ========================================================
echo.
echo [1/3] Downloading latest historical data...
cargo run --release --bin data_loader
if %errorlevel% neq 0 (
    echo [ERROR] data_loader failed!
    pause
    exit /b %errorlevel%
)

echo [2/3] Training NanoForest ML Model in nanoseconds...
cargo run --release --bin train_nano_forest
if %errorlevel% neq 0 (
    echo [ERROR] train_nano_forest failed!
    pause
    exit /b %errorlevel%
)

echo [3/3] Running Quantum Evolution (Threshold Optimization)...
cargo run --release --bin evolution_engine
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
