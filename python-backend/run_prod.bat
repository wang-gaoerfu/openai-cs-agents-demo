@echo off
set DEEPSEEK_DEV_MODE=false
python -m uvicorn api:app --reload --port 8000 