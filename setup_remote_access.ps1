# ============================================================
#  Remote Access Setup  —  treo máy nhà, SSH/Claude Code từ iPhone
#  Chạy script này với quyền Administrator
# ============================================================

Write-Host "=== [1/6] Cai OpenSSH Server ===" -ForegroundColor Cyan
$ssh = Get-WindowsCapability -Online -Name 'OpenSSH.Server*'
if ($ssh.State -ne 'Installed') {
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    Write-Host "  -> Da cai OpenSSH Server" -ForegroundColor Green
} else {
    Write-Host "  -> OpenSSH Server da co san" -ForegroundColor Green
}

Write-Host "=== [2/6] Bat & auto-start sshd service ===" -ForegroundColor Cyan
Set-Service -Name sshd -StartupType 'Automatic'
Start-Service sshd
Write-Host "  -> sshd dang chay (status: $((Get-Service sshd).Status))" -ForegroundColor Green

Write-Host "=== [3/6] Mo firewall port 22 ===" -ForegroundColor Cyan
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
    Write-Host "  -> Da tao firewall rule port 22" -ForegroundColor Green
} else {
    Write-Host "  -> Firewall rule port 22 da co" -ForegroundColor Green
}

Write-Host "=== [4/6] Dat default shell SSH = PowerShell ===" -ForegroundColor Cyan
$pwshPath = (Get-Command powershell.exe).Source
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value $pwshPath -PropertyType String -Force | Out-Null
Write-Host "  -> Default shell: $pwshPath" -ForegroundColor Green

Write-Host "=== [5/6] Tat sleep/hibernate (de may luon online) ===" -ForegroundColor Cyan
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
Write-Host "  -> Da tat standby/monitor/hibernate timeout (cam dien)" -ForegroundColor Green

Write-Host "=== [6/6] Cai Tailscale ===" -ForegroundColor Cyan
if (Test-Path 'C:\Program Files\Tailscale\tailscale.exe') {
    Write-Host "  -> Tailscale da cai san" -ForegroundColor Green
} else {
    winget install --id Tailscale.Tailscale --silent --accept-package-agreements --accept-source-agreements
    Write-Host "  -> Da cai Tailscale" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host " XONG PHAN MAY TINH. Buoc cuoi (lam thu cong):" -ForegroundColor Yellow
Write-Host " 1. Mo Tailscale tu Start Menu -> dang nhap (Google/GitHub)" -ForegroundColor White
Write-Host " 2. Ghi lai IP Tailscale cua may (dang 100.x.x.x)" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Yellow
