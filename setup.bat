@echo off
echo ============================================================
echo   Memory-Based Cognitive Twin — Setup
echo ============================================================

REM Check Python version
python --version 2>&1 | findstr /R "3\.1[12]" >nul
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.11 or 3.12 is required.
    echo         You appear to have a different version installed.
    echo         Download Python 3.11 from: https://python.org/downloads
    echo.
    pause
    exit /b 1
)

echo [OK] Python version OK
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Check for .env file
if not exist .env (
    echo [!] No .env file found. Creating one...
    copy .env.example .env
    echo.
    echo ============================================================
    echo  ACTION REQUIRED:
    echo  1. Get a FREE Groq API key from: https://console.groq.com
    echo  2. Open .env in this folder
    echo  3. Replace "your_groq_api_key_here" with your key
    echo ============================================================
) else (
    echo [OK] .env file found
)

echo.
echo ============================================================
echo   Setup complete! Run the project with:
echo     python chat.py --user alice
echo     python research_experiment.py
echo ============================================================
pause
