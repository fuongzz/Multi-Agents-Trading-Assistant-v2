@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\.venv\Scripts\python.exe"
set "LOGDIR=%~dp0logs"
set "ROOTOUT=%~dp0reports\combined_paper_trading_demo"
set "FLOW=%ROOTOUT%\flow_v2"
set "FLOWBASE=%ROOTOUT%\flow_v2_baseline"
set "FLOWTIER=%ROOTOUT%\flow_v2_tiered"
set "HOSTILE=%ROOTOUT%\hostile_combo_long"
set "CORE=%ROOTOUT%\core_mvp9_rank2"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo [%DATE% %TIME%] Combined paper trading demo started

if not exist "%PYTHON%" (
  echo Python not found: "%PYTHON%"
  exit /b 1
)

"%PYTHON%" -m scripts.wait_for_combined_eod_data --minimum-coverage 0.95 --retry-minutes 10 --retry-until 18:30
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" scripts\build_index_master.py
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m multiagents_trading_assistant.research.money_cycle.cli --input multiagents_trading_assistant/data/ohlcv_master.parquet --output data/research/money_cycle --windows 3,5,10,20,50,100,200
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m multiagents_trading_assistant.research.smart_money_trace.cli --input multiagents_trading_assistant/data/ohlcv_master.parquet --output data/research/smart_money_trace
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_production_dry_run --universe vn100 --max-candidates 10 --max-positions 2 --max-edge-rank 2 --lookback-days 14 --out-dir "%CORE%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_flow_v2_production_demo --universe vn100 --start 2020-01-01 --positions 2 --rebalance-days 10 --market-gate risk_on_or_strong_neutral --pool-filter high_rs --score-mode flow_heavy --capital 100000000 --out-dir "%FLOW%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.export_flow_v2_demo_html --demo-dir "%FLOW%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_flow_v2_production_demo --universe vn100 --start 2020-01-01 --positions 2 --rebalance-days 10 --market-gate risk_on_or_strong_neutral --pool-filter high_rs --score-mode flow_heavy --capital 100000000 --sleeve-id flow_v2_baseline --sleeve-label "Baseline fresh-signal top2" --logic-label "Flow V2 audited baseline; top2 rebalance10" --early-exit-mode none --out-dir "%FLOWBASE%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_flow_v2_production_demo --universe vn100 --start 2020-01-01 --positions 2 --rebalance-days 10 --market-gate risk_on_or_strong_neutral --pool-filter high_rs --score-mode flow_heavy --capital 100000000 --sleeve-id flow_v2_tiered --sleeve-label "Early-exit hai tang top2" --logic-label "Flow V2 audited tiered exit; top2 rebalance10" --early-exit-mode flow_momentum_tiered --out-dir "%FLOWTIER%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.run_hostile_combo_long_production_demo --start 2020-01-01 --target-exposure 1.0 --paper-enabled --out-dir "%HOSTILE%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.build_combined_paper_pnl --out-dir "%ROOTOUT%" --capital 100000000 --lot-size 100 --price-unit-multiplier 1000
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.export_combined_paper_dashboard --out-dir "%ROOTOUT%" --flow-dir "%FLOW%" --flow-baseline-dir "%FLOWBASE%" --flow-tiered-dir "%FLOWTIER%" --hostile-dir "%HOSTILE%" --core-rank2-dir "%CORE%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m scripts.update_combined_paper_realtime --out-dir "%ROOTOUT%" --html-refresh-seconds 60 --no-market-session-only --realtime-paper-entries
if errorlevel 1 exit /b %errorlevel%

echo [%DATE% %TIME%] Combined paper trading demo completed: "%ROOTOUT%\index.html"
endlocal
