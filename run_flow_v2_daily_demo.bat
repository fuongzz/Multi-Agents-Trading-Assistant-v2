@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\.venv\Scripts\python.exe"
set "LOGDIR=%~dp0logs"
set "DEMODIR=%~dp0reports\flow_v2_production_demo_live"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo [%DATE% %TIME%] Flow V2 daily demo started

if not exist "%PYTHON%" (
  echo Python not found: "%PYTHON%"
  exit /b 1
)

"%PYTHON%" scripts\update_ohlcv_daily.py
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m multiagents_trading_assistant.research.money_cycle.cli --input multiagents_trading_assistant/data/ohlcv_master.parquet --output data/research/money_cycle --windows 3,5,10,20,50,100,200
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m multiagents_trading_assistant.research.smart_money_trace.cli --input multiagents_trading_assistant/data/ohlcv_master.parquet --output data/research/smart_money_trace
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_flow_v2_production_demo --universe vn100 --start 2020-01-01 --positions 2 --rebalance-days 10 --market-gate risk_on_or_strong_neutral --pool-filter high_rs --score-mode flow_heavy --capital 1000000000 --out-dir "%DEMODIR%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.export_flow_v2_demo_html --demo-dir "%DEMODIR%"
if errorlevel 1 exit /b %errorlevel%

echo [%DATE% %TIME%] Flow V2 daily demo completed: "%DEMODIR%\index.html"
endlocal
