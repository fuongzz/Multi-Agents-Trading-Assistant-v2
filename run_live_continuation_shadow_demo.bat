@echo off
chcp 65001 > nul
setlocal

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set "VENV_PY=%~dp0UsersNC.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PYTHON=%VENV_PY%"
) else (
    set "PYTHON=python"
)

cd /d "%~dp0"

echo ============================================
echo  Live Continuation Shadow Demo
echo  - Uses current 06a/06c baseline portfolio as live paper state
echo  - Cash starts from demo Cash, open positions start from demo holdings
echo  - Full graph: trader/risk approval required before paper buy
echo  - Paper only: no broker orders
echo ============================================

"%PYTHON%" -m multiagents_trading_assistant.realtime_shadow_dual_pipeline ^
  --universe vn100 ^
  --poll-seconds 15 ^
  --continue-layered-baseline ^
  --layered-demo-dir "C:\Users\NC\Desktop\AI-Trading-Assistant\reports\layered_pipeline_demo_2026-05-17"

endlocal
