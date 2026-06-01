@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\.venv\Scripts\python.exe"
set "ROOTOUT=%~dp0reports\combined_paper_trading_demo"

if not exist "%PYTHON%" (
  echo Python not found: "%PYTHON%"
  exit /b 1
)

"%PYTHON%" -m scripts.update_combined_paper_realtime --out-dir "%ROOTOUT%" --interval-seconds 30 --html-refresh-seconds 30 --no-market-session-only --realtime-paper-entries

endlocal
