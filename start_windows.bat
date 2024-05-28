@echo off

REM Define paths
set "VENV=venv"

call .\venv\Scripts\activate.bat

call python main.py

call deactivate
pause