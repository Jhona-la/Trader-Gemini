@echo off
echo ========================================
echo  TRADER GEMINI - SPOT DASHBOARD
echo ========================================
echo Starting Spot Dashboard on port 8502...
echo Open: http://localhost:8502
echo.

.\.venv\Scripts\streamlit.exe run dashboard_spot.py --server.port 8502

pause
