@echo off
setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit

:: Ensure Miniconda/Anaconda is installed and conda is in PATH
where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not installed or not found in PATH. Please install Miniconda/Anaconda and ensure conda is in your PATH.
    goto end
)

:: Define environment name and Python version
set ENV_NAME=asuka
set PYTHON_VERSION=3.11.9

:: Deactivate any active conda environment
call conda deactivate

:: Check if the environment already exists
call conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% neq 0 (
    echo Creating conda environment %ENV_NAME% with Python %PYTHON_VERSION%
    call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
) else (
    echo Conda environment %ENV_NAME% already exists.
)

:: Activate the conda environment
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo Failed to activate conda environment %ENV_NAME%.
    goto end
)

:: Install requirements
if exist requirements.txt (
    echo Installing requirements from requirements.txt
    pip install -r requirements.txt
) else (
    echo requirements.txt not found.
    goto end
)

:: Check CUDA version and install appropriate PyTorch
set CUDA_VERSION=
for /f "tokens=2 delims==" %%i in ('wmic path win32_VideoController get DriverVersion /value') do (
    set "CUDA_VERSION=%%i"
    goto check_cuda
)

:check_cuda
if defined CUDA_VERSION (
    if %CUDA_VERSION% geq 12.3 (
        echo CUDA version detected: %CUDA_VERSION%. Installing PyTorch for CUDA 12.3.
        call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if %CUDA_VERSION% geq 11.8 (
        echo CUDA version detected: %CUDA_VERSION%. Installing PyTorch for CUDA 11.8.
        call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo CUDA version detected: %CUDA_VERSION%. Installing CPU version of PyTorch.
        call pip install torch torchvision torchaudio
    )
) else (
    echo No CUDA version detected. Installing CPU version of PyTorch.
    call pip install torch torchvision torchaudio
)

call ollama create Asuka -f models/modelfile.md

:: Run main.py
if exist main.py (
    echo Running main.py
    python main.py
) else (
    echo main.py not found.
    goto end
)

:: Deactivate conda environment before exit
call conda deactivate

:end
pause
endlocal