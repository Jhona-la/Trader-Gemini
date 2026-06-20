@echo off
echo Running Holographic Audit Scripts...

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: .venv not found. Using global python.
)

echo.
echo Running Silent Failures Audit...
python scratch\audit_silent_failures.py
if errorlevel 1 echo Error running audit_silent_failures.py

echo.
echo Running Features Audit...
python scratch\audit_features.py
if errorlevel 1 echo Error running audit_features.py

echo.
echo Audit scripts completed. Please check scratch\silent_failures.json and scratch\feature_audit.json
pause
