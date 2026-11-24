@echo off
setlocal
cd /d "%~dp0"
if exist ".venv-fin\Scripts\python.exe" goto launch
call setup.bat
if errorlevel 1 exit /b 1

:launch
".venv-fin\Scripts\python.exe" -c "import streamlit, pandas, numpy, plotly, openpyxl, duckdb, sklearn, mlflow" 2>nul
if not errorlevel 1 goto run
call setup.bat
if errorlevel 1 exit /b 1

:run
echo Open the local URL printed by Streamlit.
".venv-fin\Scripts\python.exe" -m streamlit run app.py

