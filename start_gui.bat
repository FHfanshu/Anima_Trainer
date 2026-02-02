@echo off
chcp 65001 >nul
title Anima LoRA Trainer GUI - 一键启动
echo.
echo ============================================================
echo     Anima LoRA Trainer GUI - 一键启动脚本
echo ============================================================
echo.
echo  此脚本将自动：
echo    1. 检查 Python 环境
echo    2. 安装缺失的 Python 依赖
echo    3. 检查 Node.js 环境
echo    4. 安装前端依赖 (npm install)
echo    5. 构建前端 (npm run build)
echo    6. 启动服务并打开浏览器
echo.
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python 检测通过
echo.
echo 正在启动...
echo.

REM 运行 Python 启动脚本
python "%~dp0start_gui.py"

if errorlevel 1 (
    echo.
    echo [错误] 启动失败
    pause
)
