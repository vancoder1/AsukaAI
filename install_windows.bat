@echo off

call ollama create Asuka -f models/modelfile.md

REM Define paths
set "VENV=venv"

REM Check if the virtual environment folder exists
IF NOT EXIST %VENV% (
    call pip install virtualenv
    echo Creating virtual environment...
    call python -m venv %VENV%
)

REM Install required packages
IF EXIST requirements.txt (
    echo Installing required packages...
    pip install -r requirements.txt
)

call python main.py

call deactivate
pause