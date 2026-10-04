@echo off
setlocal
title musicviz - instalacion
cd /d "%~dp0"
echo ============================================================
echo   musicviz - instalacion en Windows
echo ============================================================
echo.

rem ---- 1) Python
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
    echo [X] No se encontro Python. Se abrira la pagina de descarga.
    echo     Al instalarlo marca "Add python.exe to PATH" y vuelve a ejecutar este archivo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('%PY% --version 2^>^&1') do echo [OK] %%v

rem ---- 2) ffmpeg
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [..] ffmpeg no esta instalado. Intentando instalarlo con winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    echo.
    echo     Si winget lo instalo, cierra esta ventana, abre una nueva y ejecuta instalar.bat otra vez
    echo     para que el sistema reconozca ffmpeg. Si no, descargalo de https://www.gyan.dev/ffmpeg/builds/
    echo     y agrega su carpeta bin al PATH.
    pause
    exit /b 1
) else (
    echo [OK] ffmpeg encontrado
)

rem ---- 3) entorno virtual + dependencias
if not exist ".venv\Scripts\python.exe" (
    echo [..] Creando entorno virtual...
    %PY% -m venv .venv || ( echo [X] No se pudo crear el entorno virtual & pause & exit /b 1 )
)
echo [..] Instalando musicviz y sus dependencias (puede tardar unos minutos)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -e . || ( echo [X] Fallo la instalacion de dependencias & pause & exit /b 1 )
".venv\Scripts\python.exe" -m pip install sounddevice >nul 2>nul

rem ---- 4) comprobacion
echo.
".venv\Scripts\musicviz.exe" check
echo.
echo ============================================================
echo   Listo. Abre la interfaz con doble clic en musicviz-gui.bat
echo ============================================================
pause
endlocal
