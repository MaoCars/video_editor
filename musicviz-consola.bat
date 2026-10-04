@echo off
title musicviz
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Primero ejecuta instalar.bat
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Entorno de musicviz activo. Ejemplos:
echo   musicviz gui
echo   musicviz init mi_video.yaml --audio "C:\Musica\cancion.mp3" --preset trap_nation
echo   musicviz render mi_video.yaml
echo.
cmd /k
