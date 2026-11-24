@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-fin\Scripts\mlflow.exe" call setup.bat
if errorlevel 1 exit /b 1
set MLFLOW_DISABLE_TELEMETRY=true
".venv-fin\Scripts\python.exe" -m finance.tracking
