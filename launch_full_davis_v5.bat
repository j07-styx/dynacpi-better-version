@echo off
setlocal
title DynaCPI v5 Full DAVIS Demo
cd /d "%~dp0"
set "PY=%USERPROFILE%\dynacpi_v4\.venv\Scripts\python.exe"
if defined DYNACPI_PYTHON set "PY=%DYNACPI_PYTHON%"
if not exist "%PY%" (
  echo No Python environment found at "%PY%"
  echo Set DYNACPI_PYTHON to your Python interpreter with Torch, Flask and NumPy installed.
  pause
  exit /b 1
)
if not exist "%~dp0full_davis_demo_20261109\train_v5\RESULTS.json" (
  echo Trained full-DAVIS v5 checkpoints and result files are missing.
  echo Check that you are on the original machine with the local experiment data.
  pause
  exit /b 1
)
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=2"
echo Starting the local DynaCPI research console.
echo Open http://127.0.0.1:8782/ in Microsoft Edge.
"%PY%" -u "%~dp0demo_full_davis_v5.py"
if errorlevel 1 pause
