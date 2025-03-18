@echo off
setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit

:: Set UTF-8 code page [[1]]
chcp 65001 >nul

:: Install Header
echo ┌───────────────────────────────────┐
echo │          Asuka Installer          │
echo └───────────────────────────────────┘

:: Verify Conda installation
where conda >nul 2>nul || (
    echo [ERROR] Conda not found in PATH
    echo [HELP] Download Miniconda from:
    echo https://docs.conda.io/en/latest/miniconda.html
    goto end
)

:: Environment Setup
set "ENV_NAME=asuka"
set "PYTHON_VERSION=3.11.9"

:: Remove existing environment to ensure clean install
call conda deactivate 2>nul
call conda env remove -n %ENV_NAME% -y

:: Create environment
echo Creating environment with Python %PYTHON_VERSION%
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y || (
    echo [ERROR] Environment creation failed
    goto end
)

:: Activate environment and install dependencies
call conda activate %ENV_NAME% || (
    echo [ERROR] Failed to activate environment
    goto end
)

:: Install base requirements
if exist "%~dp0requirements.txt" (
    echo Installing core dependencies...
    pip install -r "%~dp0requirements.txt" || (
        echo [ERROR] Core dependency installation failed
        goto end
    )
) else (
    echo [ERROR] requirements.txt not found
    goto end
)

:: CUDA Detection and PyTorch Installation
echo Checking NVIDIA CUDA compatibility...
set "CUDA_VERSION="
nvcc --version 2>nul | find "release" >nul && (
    for /f "tokens=2 delims=," %%v in ('nvcc --version ^| find "release"') do (
        set "CUDA_VERSION=%%v"
    )
)

if defined CUDA_VERSION (
    echo Detected CUDA %CUDA_VERSION%
    if "%CUDA_VERSION:~1,4%" geq "12.1" (
        echo Installing PyTorch with CUDA 12.1 support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if "%CUDA_VERSION:~1,4%" geq "11.8" (
        echo Installing PyTorch with CUDA 11.8 support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo Installing CPU version of PyTorch...
        pip install torch torchvision torchaudio
    )
) else (
    echo No NVIDIA CUDA detected. Installing CPU version...
    pip install torch torchvision torchaudio
)

:: Verify critical dependencies
echo Verifying installation...
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "from RealtimeSTT import AudioToTextRecorder; print('RealtimeSTT verified')"

:: Ollama Check
ollama --version >nul 2>&1 || (
    echo [WARNING] Ollama not detected - some features may not work
    echo Download from: https://ollama.com/download
)

:end
echo Installation complete! Run start_windows.bat to launch
pause