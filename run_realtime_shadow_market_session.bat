@echo off
chcp 65001 > nul
setlocal

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set SESSION=%1
if "%SESSION%"=="" set SESSION=MORNING

set SYMBOL=%2
if "%SYMBOL%"=="" set SYMBOL=VN100

set STOP_AT=11:30
if /I "%SESSION%"=="AFTERNOON" set STOP_AT=14:45
if /I "%SESSION%"=="PM" set STOP_AT=14:45

set "VENV_PY=%~dp0UsersNC.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PYTHON=%VENV_PY%"
) else (
    set "PYTHON=python"
)

set MODE_ARGS=%SYMBOL%
if /I "%SYMBOL%"=="VN100" set MODE_ARGS=--universe vn100
if /I "%SYMBOL%"=="VN30" set MODE_ARGS=--universe vn30

cd /d "%~dp0"

echo [%date% %time%] Realtime shadow %SESSION% session started for %SYMBOL%, stop-at %STOP_AT%.
"%PYTHON%" -m multiagents_trading_assistant.realtime_shadow_dual_pipeline %MODE_ARGS% --poll-seconds 15 --stop-at %STOP_AT%
echo [%date% %time%] Realtime shadow %SESSION% session finished.

endlocal
