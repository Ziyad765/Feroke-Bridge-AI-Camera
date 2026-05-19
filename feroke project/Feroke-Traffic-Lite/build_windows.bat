@echo off
REM ════════════════════════════════════════════════════════════════
REM  FEROKE VANGUARD — Windows .EXE Builder
REM  Run this file from inside Feroke-Traffic-Lite\
REM ════════════════════════════════════════════════════════════════

echo.
echo =========================================
echo  FEROKE VANGUARD — BUILD SYSTEM
echo =========================================
echo.

REM Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found on PATH. Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo [1/4] Installing PyInstaller...
pip install pyinstaller --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyInstaller.
    pause
    exit /b 1
)

echo [2/4] Installing all project dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo [3/4] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo [4/4] Building FerokeVanguard.exe (this may take 3-8 minutes)...
pyinstaller feroke.spec

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build FAILED. See errors above.
    pause
    exit /b 1
)

echo.
echo =========================================
echo  BUILD SUCCESS!
echo =========================================
echo  Output: dist\FerokeVanguard\FerokeVanguard.exe
echo.
echo  HOW TO DEPLOY:
echo    1. Copy the entire  dist\FerokeVanguard\  folder to any Windows PC.
echo    2. Double-click FerokeVanguard.exe to launch.
echo    3. No Python installation needed on target machine.
echo.
echo  IMPORTANT — Edit config.py BEFORE building if you need to change
echo  the NVR IP, camera channels, or model path.
echo =========================================
echo.
pause
