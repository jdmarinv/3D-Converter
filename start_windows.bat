@echo off
setlocal enabledelayedexpansion
title 2D to 3D Studio - Offline Stereoscopic Engine
cd /d "%~dp0"

echo ==========================================================
echo          2D TO 3D STUDIO - OFFLINE CONVERTER
echo ==========================================================

:: 1. Check for Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
)

:: 2. Find Python Executable
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PYTHON_CMD=py -3
    ) else (
        echo.
        echo [ERROR] Python was not found on your system!
        echo Please download and install Python 3.10, 3.11, or 3.12 from:
        echo https://www.python.org/downloads/
        echo *Make sure to check "Add python.exe to PATH" during installation.*
        echo.
        pause
        exit /b 1
    )
)

:: 3. Launch GUI Studio
echo [INFO] Starting 2D to 3D Studio Web GUI...
%PYTHON_CMD% gui.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    echo If dependencies are missing, run:
    echo     pip install -r requirements.txt
    echo.
    pause
)
