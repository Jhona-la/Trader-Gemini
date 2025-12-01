@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
echo 🔵 STARTING TRADER GEMINI - FUTURES MODE
python main.py --mode futures
pause
