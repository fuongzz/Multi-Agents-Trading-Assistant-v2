@echo off
chcp 65001 > nul
setlocal

set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=VN100

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_realtime_shadow_scheduler.ps1" -Universe "%SYMBOL%"

endlocal
