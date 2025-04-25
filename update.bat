@echo off
setlocal enabledelayedexpansion
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit

:: Set UTF-8 code page [[1]]
chcp 65001 >nul

:: Update Header
echo ┌─────────────────────────────────┐
echo │          Asuka Updater          │
echo └─────────────────────────────────┘

:: Git Validation
where git >nul 2>nul || (
    echo [ERROR] Git not found
    echo Install from: https://git-scm.com/
    goto end
)

:: Repository Checks
cd /d "%~dp0" || (
    echo [ERROR] Failed to locate project directory
    goto end
)

:: Smart Update Process [[9]]
echo Checking for updates...
git fetch origin
for /f "tokens=*" %%b in ('git rev-parse --abbrev-ref HEAD') do set BRANCH=%%b

git status -uno | find "up to date" >nul
if %errorlevel% neq 0 (
    echo Pulling latest changes from %BRANCH%...
    git pull origin %BRANCH% || (
        echo [ERROR] Update failed
        goto end
    )
    
    :: Dependency Update
    if exist requirements.txt (
        echo Updating dependencies...
        pip install -r requirements.txt --upgrade
    )
    
    :: Environment Check
    conda list python | find "%PYTHON_VERSION%" >nul || (
        echo [WARNING] Python version changed
        echo Run install_windows.bat to update environment
    )
) else (
    echo Already up to date
)

:end
pause