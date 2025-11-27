@echo off
echo ========================================
echo  TRADER GEMINI - FUTURES DASHBOARD
echo ========================================
echo Starting Futures Dashboard on port 8501...
echo Open: http://localhost:8501
echo.

.\.venv\Scripts\streamlit.exe run dashboard/app.py

pause
