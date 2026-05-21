@echo off
chcp 65001 > nul
title Intraday Trade MVP Monitor

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=VN100

set MODE_ARGS=%SYMBOL%
if /I "%SYMBOL%"=="VN100" set MODE_ARGS=--universe vn100
if /I "%SYMBOL%"=="VN30" set MODE_ARGS=--universe vn30

echo ============================================
echo  Intraday Trade MVP Monitor - %SYMBOL%
echo  Press Ctrl+C to stop
echo ============================================

:loop
echo [%date% %time%] Starting intraday MVP monitor for %SYMBOL%...
python -m multiagents_trading_assistant.intraday_trade_mvp_monitor %MODE_ARGS% --poll-seconds 15 --discord

echo.
echo [%date% %time%] Monitor stopped. Restarting in 15 seconds...
echo (Press Ctrl+C to cancel)
timeout /t 15 /nobreak > nul
goto loop
