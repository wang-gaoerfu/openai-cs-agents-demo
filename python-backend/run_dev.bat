@echo off
set DEEPSEEK_DEV_MODE=true
python -m uvicorn api:app --reload --port 8000 