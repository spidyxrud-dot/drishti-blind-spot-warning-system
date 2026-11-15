@echo off
setlocal
cd /d "%~dp0"
if exist ".python\python.exe" (
    ".python\python.exe" main.py %*
    exit /b %errorlevel%
)
where python >nul 2>nul
if %errorlevel%==0 (
    python main.py %*
    exit /b %errorlevel%
)
echo Python was not found. Install Python 3.10-3.12 or use the bundled .python runtime.
pause
