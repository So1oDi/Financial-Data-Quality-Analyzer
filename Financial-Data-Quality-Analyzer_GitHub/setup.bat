@echo off
setlocal
cd /d "%~dp0"
if exist ".venv-fin\Scripts\python.exe" goto install
set "LOCAL_PY=%LOCALAPPDATA%\Python\pythoncore-3.15-64\python.exe"
if exist "%LOCAL_PY%" goto localpy
py -3 -m venv .venv-fin
if not errorlevel 1 goto install
python -m venv .venv-fin
if errorlevel 1 goto failed
goto install

:localpy
"%LOCAL_PY%" -m venv .venv-fin
if errorlevel 1 goto failed

:install
".venv-fin\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto failed
".venv-fin\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
echo Setup complete. Run start.bat.
exit /b 0

:failed
echo Setup failed. Check that CPython 3.12-3.15 x64 and access to PyPI are available.
exit /b 1

