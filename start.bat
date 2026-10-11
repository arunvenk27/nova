@echo off
REM ================================================================
REM  NOVA dashboard - double-click to start (Windows)
REM  Optional: start.bat 8080   (use another port if 8000 is busy)
REM ================================================================
setlocal
cd /d "%~dp0"
set PORT=%1
if "%PORT%"=="" set PORT=8000
title NOVA dashboard (port %PORT%)

echo.
echo   NOVA - Natural Observation and Vegetative Analysis
echo   ==================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo   [!] Python was not found.
  echo       Install Python 3.10 or newer from https://www.python.org/downloads/
  echo       and tick "Add python.exe to PATH" during installation.
  echo.
  pause
  exit /b 1
)

python -c "import fastapi, uvicorn, numpy, pandas, scipy, sklearn, matplotlib" >nul 2>&1
if errorlevel 1 (
  echo   Installing the required packages - first run only, needs internet...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo   [!] Package installation failed. See README.md - Troubleshooting.
    pause
    exit /b 1
  )
)

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
  echo   [!] Python packages are installed but cannot load.
  echo       If the error mentions "Application Control policy", turn off
  echo       Smart App Control - see README.md - Troubleshooting.
  python -c "import numpy"
  pause
  exit /b 1
)

echo   Starting NOVA at http://localhost:%PORT%
echo   The dashboard opens in your browser in a few seconds.
echo   Keep this window open while presenting. Close it to stop.
echo.
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:%PORT%"
python -m uvicorn backend.app:app --host 127.0.0.1 --port %PORT%

echo.
echo   The server stopped. If you see "address already in use",
echo   run:  start.bat 8080
pause
