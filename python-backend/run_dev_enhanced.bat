@echo off
chcp 65001 > nul
echo ========================================
echo 启动增强版开发模式
echo ========================================

REM 设置环境变量
set DEEPSEEK_DEV_MODE=true
set PYTHONPATH=%PYTHONPATH%;%CD%
set PYTHONIOENCODING=utf-8

REM 显示环境变量状态
echo 环境变量设置:
echo DEEPSEEK_DEV_MODE=%DEEPSEEK_DEV_MODE%
echo PYTHONPATH=%PYTHONPATH%
echo PYTHONIOENCODING=%PYTHONIOENCODING%

REM 启动服务器
echo ========================================
echo 启动服务器...
echo ========================================
python -m uvicorn api:app --reload --port 8000 