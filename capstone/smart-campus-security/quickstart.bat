@echo off
REM Quick start script for Smart Campus Security & Attendance 2.0
REM Windows batch file

echo ========================================
echo Smart Campus Security - Quick Start
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages may have failed to install
)

echo.
echo [2/4] Generating alert sound...
python assets/generate_sound.py
if errorlevel 1 (
    echo WARNING: Alert sound generation failed
)

echo.
echo [3/4] Initializing database...
python -c "from db import Database; Database()"
if errorlevel 1 (
    echo ERROR: Database initialization failed
    pause
    exit /b 1
)

echo.
echo [4/4] Seeding demo data...
set /p seed="Generate demo data? (Y/n): "
if /i "%seed%"=="n" goto skip_seed
python tests/seed_data.py
:skip_seed

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application, run:
echo   streamlit run main.py
echo.
echo Then open: http://localhost:8501
echo.
pause
