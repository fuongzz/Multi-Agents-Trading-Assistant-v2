param(
    [string]$At = "15:55"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $Root "run_combined_paper_trading_demo.bat"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogPath = Join-Path $LogDir "combined_paper_trading_demo.log"
$Command = "/c `"`"$Runner`" >> `"$LogPath`" 2>&1`""
$Action = New-ScheduledTaskAction -Execute "$env:ComSpec" -Argument $Command -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "AI-Trading-Combined-Paper-Demo" `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "AI Trading Assistant combined paper dashboard for MVP p5, MVP p4, and Flow V2." `
    -Force | Out-Null

Write-Host "Registered scheduled task:"
Write-Host "  AI-Trading-Combined-Paper-Demo Mon-Fri $At"
Write-Host "Output:"
Write-Host "  reports\combined_paper_trading_demo\index.html"
Write-Host "Log:"
Write-Host "  $LogPath"
