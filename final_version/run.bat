@echo off
title COS30018 Fall Detection System
echo ============================================
echo   COS30018 - Fall Detection System
echo   Swinburne University of Technology
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ from python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check if dependencies installed
python -c "import torch, cv2, ultralytics, mediapipe, PyQt5" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing required dependencies...
    echo        This may take a few minutes on first run.
    echo.
    pip install -r "%~dp0requirements.txt"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Dependencies installed successfully.
    echo.
)

echo [INFO] Starting Fall Detection GUI...
echo        Close this window to stop the application.
echo.
python "%~dp0GUI_fall_detection.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
