@echo off
title DeepSeek Search Assistant

rem Switch to project root directory (parent of deepseek_search/)
set "PROJECT_DIR=%~dp0.."
pushd "%PROJECT_DIR%"

set "VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [Error] Virtual environment not found at: .venv
    echo.
    echo Please run the following commands in the project directory:
    echo   uv venv
    echo   uv pip install -r "%~dp0requirements.txt"
    popd
    pause
    exit /b 1
)

echo Project Directory: %PROJECT_DIR%
echo.

"%VENV_PYTHON%" -m deepseek_search.main %*
if %errorlevel% neq 0 (
    echo.
    echo [Error] Failed to start.
    pause
)

popd
