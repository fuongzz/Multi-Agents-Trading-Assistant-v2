@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\.venv\Scripts\python.exe"
set "LOGDIR=%~dp0logs"
set "ROOTOUT=%~dp0reports\combined_paper_trading_demo"
set "MVP5=%ROOTOUT%\mvp_p5"
set "MVP4=%ROOTOUT%\mvp_p4"
set "FLOW=%ROOTOUT%\flow_v2"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo [%DATE% %TIME%] Combined paper trading demo started

if not exist "%PYTHON%" (
  echo Python not found: "%PYTHON%"
  exit /b 1
)

"%PYTHON%" scripts\update_ohlcv_daily.py
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m multiagents_trading_assistant.research.money_cycle.cli --input multiagents_trading_assistant/data/ohlcv_master.parquet --output data/research/money_cycle --windows 3,5,10,20,50,100,200
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_production_dry_run --universe vn100 --max-candidates 10 --max-positions 5 --lookback-days 14 --out-dir "%MVP5%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_production_dry_run --universe vn100 --max-candidates 10 --max-positions 4 --lookback-days 14 --out-dir "%MVP4%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_flow_v2_production_demo --universe vn100 --start 2020-01-01 --positions 2 --rebalance-days 20 --market-gate risk_on_or_strong_neutral --pool-filter clean_flow --score-mode sector_heavy --capital 100000000 --out-dir "%FLOW%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.export_flow_v2_demo_html --demo-dir "%FLOW%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.build_combined_paper_pnl --out-dir "%ROOTOUT%" --capital 100000000 --lot-size 100 --price-unit-multiplier 1000
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.export_combined_paper_dashboard --out-dir "%ROOTOUT%" --mvp-p5-dir "%MVP5%" --mvp-p4-dir "%MVP4%" --flow-dir "%FLOW%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.update_combined_paper_realtime --out-dir "%ROOTOUT%" --html-refresh-seconds 60 --no-market-session-only
if errorlevel 1 exit /b %errorlevel%

echo [%DATE% %TIME%] Combined paper trading demo completed: "%ROOTOUT%\index.html"
endlocal
