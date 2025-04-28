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
set "PYTHON_VERSION=3.12.9"

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

:: PyTorch Installation Choice
echo Select PyTorch version to install:
echo 1. CUDA 12.3 (Requires compatible NVIDIA GPU and drivers)
echo 2. CUDA 11.8 (Requires compatible NVIDIA GPU and drivers)
echo 3. CPU Only
set /p TORCH_CHOICE="Enter choice (1, 2, or 3): "

if "%TORCH_CHOICE%"=="1" (
    echo Installing PyTorch for CUDA 12.3...
    call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || (
        echo [ERROR] PyTorch CUDA 12.3 installation failed
        goto end
    )
) else if "%TORCH_CHOICE%"=="2" (
    echo Installing PyTorch for CUDA 11.8...
    call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || (
        echo [ERROR] PyTorch CUDA 11.8 installation failed
        goto end
    )
) else if "%TORCH_CHOICE%"=="3" (
    echo Installing CPU version of PyTorch...
    call pip install torch torchvision torchaudio || (
        echo [ERROR] PyTorch CPU installation failed
        goto end
    )
) else (
    echo Invalid choice. Please run the installer again.
    goto end
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