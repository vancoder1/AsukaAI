@echo off

REM Define paths
set "VENV=venv"

call python -m venv %VENV%

call python main.py

call deactivate
pause