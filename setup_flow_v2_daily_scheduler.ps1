param(
    [string]$At = "15:50"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $Root "run_flow_v2_daily_demo.bat"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogPath = Join-Path $LogDir "flow_v2_daily_demo.log"
$Command = "/c `"`"$Runner`" >> `"$LogPath`" 2>&1`""
$Action = New-ScheduledTaskAction -Execute "$env:ComSpec" -Argument $Command -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "AI-Trading-FlowV2-Daily-Demo" `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "AI Trading Assistant Flow V2 daily production demo after market close." `
    -Force | Out-Null

Write-Host "Registered scheduled task:"
Write-Host "  AI-Trading-FlowV2-Daily-Demo Mon-Fri $At"
Write-Host "Output:"
Write-Host "  reports\flow_v2_production_demo_live\index.html"
Write-Host "Log:"
Write-Host "  $LogPath"
