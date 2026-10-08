@echo off
title musicviz - crear ejecutable
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Primero ejecuta instalar.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -q pyinstaller
".venv\Scripts\python.exe" packaging\build.py --ffmpeg --zip
echo.
echo El ejecutable esta en dist\musicviz\musicviz-gui.exe (y el ZIP en dist\)
pause
