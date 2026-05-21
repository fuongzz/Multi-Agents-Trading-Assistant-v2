@echo off
chcp 65001 > nul
title Realtime Money Flow Monitor

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=HPG

set MODE_ARGS=%SYMBOL%
if /I "%SYMBOL%"=="VN100" set MODE_ARGS=--universe vn100
if /I "%SYMBOL%"=="VN30" set MODE_ARGS=--universe vn30

echo ============================================
echo  Realtime Money Flow Monitor - %SYMBOL%
echo  Press Ctrl+C to stop
echo ============================================

:loop
echo [%date% %time%] Starting monitor for %SYMBOL%...
python -m multiagents_trading_assistant.intraday_money_flow_monitor %MODE_ARGS% --poll-seconds 5 --window-minutes 5 --flow-threshold-b 1.0 --net-threshold-b 0.5 --volume-threshold 300000 --summary-seconds 30 --top-n 10 --discord

echo.
echo [%date% %time%] Monitor stopped. Restarting in 15 seconds...
echo (Press Ctrl+C to cancel)
timeout /t 15 /nobreak > nul
goto loop
