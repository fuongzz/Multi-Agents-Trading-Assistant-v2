@echo off
chcp 65001 > nul
title Realtime Shadow Dual Pipeline

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=VN100

set "VENV_PY=%~dp0UsersNC.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PYTHON=%VENV_PY%"
) else (
    set "PYTHON=python"
)

set MODE_ARGS=%SYMBOL%
if /I "%SYMBOL%"=="VN100" set MODE_ARGS=--universe vn100
if /I "%SYMBOL%"=="VN30" set MODE_ARGS=--universe vn30

echo ============================================
echo  Realtime Shadow Dual Pipeline - %SYMBOL%
echo  Dashboards: reports\realtime_shadow
echo  Layer Demo: reports\layered_pipeline_demo_2026-05-17\index.html
echo  Press Ctrl+C to stop
echo ============================================

:loop
echo [%date% %time%] Starting dual paper runner for %SYMBOL%...
"%PYTHON%" -m multiagents_trading_assistant.realtime_shadow_dual_pipeline %MODE_ARGS% --poll-seconds 15

echo.
echo [%date% %time%] Runner stopped. Restarting in 15 seconds...
echo (Press Ctrl+C to cancel)
timeout /t 15 /nobreak > nul
goto loop
