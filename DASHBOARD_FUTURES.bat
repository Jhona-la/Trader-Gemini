@echo off
setlocal
cd /d "%~dp0"
title [DASHBOARD] TRADER GEMINI - MONITOR
color 09

echo.
echo 📊 STARTING TRADER GEMINI DASHBOARD
echo.

call .venv\Scripts\activate.bat

streamlit run dashboard/app.py --server.port 8501 --server.headless true

echo.
echo [INFO] Dashboard closed.
pause
