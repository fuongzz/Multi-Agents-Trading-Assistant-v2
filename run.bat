@echo off
chcp 65001 > nul
title AI Trading Assistant

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ============================================
echo  AI Trading Assistant - Starting...
echo  Press Ctrl+C to stop
echo ============================================

:loop
echo [%date% %time%] Pulling latest code from GitHub...
git pull origin main

echo [%date% %time%] Starting scheduler...
python -m multiagents_trading_assistant.main --schedule

echo.
echo [%date% %time%] Process stopped. Restarting in 15 seconds...
echo (Press Ctrl+C to cancel)
timeout /t 15 /nobreak > nul
goto loop
