@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" -c "import PySide6, cv2, numpy, PIL, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo [错误] 环境未安装，请先运行 install.bat。
    pause
    exit /b 1
)

if not exist "models\sam_vit_b_encoder.onnx" (
    echo [错误] 缺少 SAM 模型，请先运行 install.bat。
    pause
    exit /b 1
)

if not exist "output_interactive" mkdir "output_interactive"

if "%~1"=="" (
    "%PY%" interactive_gui.py
) else (
    "%PY%" interactive_gui.py "%~1"
)
if errorlevel 1 (
    echo.
    echo [错误] 交互程序异常退出。
    pause
)
endlocal
