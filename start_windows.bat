@echo off
setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit

:: Set UTF-8 code page [[1]]
chcp 65001 >nul

:: Launch Header
echo ┌──────────────────────────────────┐
echo │          Asuka Launcher          │
echo └──────────────────────────────────┘

:: Verify Conda installation
where conda >nul 2>nul || (
    echo [ERROR] Conda not found - run install_windows.bat first
    goto end
)

:: Environment Activation
set "ENV_NAME=asuka"
call conda activate %ENV_NAME% || (
    echo [ERROR] Failed to activate environment
    echo [HELP] Run install_windows.bat to setup
    goto end
)

:: Verify working directory
cd /d "%~dp0" || (
    echo [ERROR] Failed to locate project directory
    goto end
)

:: Dependency Check
echo Verifying dependencies...
python -c "import torch; import RealtimeSTT" 2>nul || (
    echo [ERROR] Missing dependencies - run install_windows.bat
    goto end
)

:: GPU Check
echo Checking GPU availability...
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

:: Launch Application
echo Starting Asuka...
python "%~dp0main.py" %*
if %errorlevel% neq 0 (
    echo [ERROR] Application crashed with code %errorlevel%
    echo [HELP] Check requirements and environment setup
)

:end
call conda deactivate
pause