@echo off
chcp 65001 > nul
title Realtime Money Flow Monitor

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=HPG

echo ============================================
echo  Realtime Money Flow Monitor - %SYMBOL%
echo  Press Ctrl+C to stop
echo ============================================

:loop
echo [%date% %time%] Starting monitor for %SYMBOL%...
python -m multiagents_trading_assistant.intraday_money_flow_monitor %SYMBOL% --poll-seconds 5 --window-minutes 5 --flow-threshold-b 1.0 --net-threshold-b 0.5 --volume-threshold 300000 --discord

echo.
echo [%date% %time%] Monitor stopped. Restarting in 15 seconds...
echo (Press Ctrl+C to cancel)
timeout /t 15 /nobreak > nul
goto loop
