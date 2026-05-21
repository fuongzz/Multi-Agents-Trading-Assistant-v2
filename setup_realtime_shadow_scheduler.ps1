param(
    [ValidateSet("VN100", "VN30")]
    [string]$Universe = "VN100"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $Root "run_realtime_shadow_market_session.bat"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Register-ShadowTask {
    param(
        [string]$Name,
        [string]$Session,
        [string]$At,
        [string]$LogName
    )

    $LogPath = Join-Path $LogDir $LogName
    $Command = "/c `"`"$Runner`" $Session $Universe >> `"$LogPath`" 2>&1`""
    $Action = New-ScheduledTaskAction -Execute "$env:ComSpec" -Argument $Command -WorkingDirectory $Root
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

    Register-ScheduledTask `
        -TaskName $Name `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "AI Trading Assistant realtime shadow pipeline $Session session ($Universe)." `
        -Force | Out-Null
}

Register-ShadowTask -Name "AI-Trading-Shadow-Morning" -Session "MORNING" -At "09:00" -LogName "realtime_shadow_morning.log"
Register-ShadowTask -Name "AI-Trading-Shadow-Afternoon" -Session "AFTERNOON" -At "13:00" -LogName "realtime_shadow_afternoon.log"

Write-Host "Registered scheduled tasks:"
Write-Host "  AI-Trading-Shadow-Morning   Mon-Fri 09:00, stops at 11:30"
Write-Host "  AI-Trading-Shadow-Afternoon Mon-Fri 13:00, stops at 14:45"
Write-Host "Logs:"
Write-Host "  $LogDir"
