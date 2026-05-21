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
echo  Auto Market Data Realtime Daemon
echo  - Trong phien: live holdings + VN100 price snapshot
echo  - Ngoai phien: dung gia dong cua moi nhat
echo  - Sau phien: update OHLCV/VNINDEX/research, retry den 23:00 neu chua co close hom nay
echo  Dashboard: reports\layered_pipeline_demo_2026-05-17\index.html
echo ============================================

"%PYTHON%" scripts\auto_update_data_system.py ^
  --daily-time 14:55 ^
  --eod-retry-until 23:00 ^
  --daily-retry-minutes 10 ^
  --poll-seconds 30 ^
  --live-interval-seconds 15 ^
  --live-market-interval-seconds 60 ^
  --demo-dir "C:\Users\NC\Desktop\AI-Trading-Assistant\reports\layered_pipeline_demo_2026-05-17" ^
  --layered-demo-dir "C:\Users\NC\Desktop\AI-Trading-Assistant\reports\layered_pipeline_demo_2026-05-17" ^
  --skip-demo ^
  --allow-quality-warn

endlocal
