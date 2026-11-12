@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "BASE_PY=python"
%BASE_PY% -c "import sys" >nul 2>&1
if errorlevel 1 set "BASE_PY=py -3"
%BASE_PY% -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [错误] 找不到 Python 3.10 - 3.14（64 位）。
    pause
    exit /b 1
)

%BASE_PY% -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info<(3,15) else 1)"
if errorlevel 1 (
    echo [错误] Python 版本必须是 3.10 - 3.14。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 创建虚拟环境...
    %BASE_PY% -m venv .venv
)
set "PY=%CD%\.venv\Scripts\python.exe"

echo [2/4] 安装依赖...
"%PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
"%PY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo [错误] 依赖安装失败。
    pause
    exit /b 1
)

echo [3/4] 下载并校验 SAM 交互模型...
"%PY%" download_models.py
if errorlevel 1 (
    echo [错误] 模型下载失败。
    pause
    exit /b 1
)

echo [4/4] 执行自检...
"%PY%" setup_check.py
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo 安装完成。现在可以运行 run_interactive.bat。
pause
endlocal
