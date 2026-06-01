# ============================================================
#  Tao SSH key cho dang nhap tu iPhone (Termius)
#  Tu phat hien admin/standard, dat public key dung cho,
#  set ACL chuan Windows OpenSSH, in ra private key.
# ============================================================

$ErrorActionPreference = 'Stop'

# --- 1. Xac dinh NC co phai admin ---
$adminMembers = Get-LocalGroupMember -Group 'Administrators' | Select-Object -ExpandProperty Name
$isAdmin = $false
foreach ($m in $adminMembers) {
    if ($m -like '*\NC' -or $m -eq 'NC') { $isAdmin = $true }
}
Write-Host "NC is admin: $isAdmin" -ForegroundColor Cyan

# --- 2. Tao cap khoa ed25519 (khong passphrase) ---
$keyDir  = "C:\Users\NC\.ssh"
$keyPath = Join-Path $keyDir "iphone_termius"
if (-not (Test-Path $keyDir)) { New-Item -ItemType Directory -Path $keyDir -Force | Out-Null }
if (Test-Path $keyPath) { Remove-Item "$keyPath*" -Force }

ssh-keygen -t ed25519 -f $keyPath -N '""' -C "iphone-termius" | Out-Null
Write-Host "Da tao keypair tai $keyPath" -ForegroundColor Green

$pubKey = Get-Content "$keyPath.pub" -Raw

# --- 3. Dat public key vao dung file authorized_keys ---
if ($isAdmin) {
    $authFile = "C:\ProgramData\ssh\administrators_authorized_keys"
    Add-Content -Path $authFile -Value $pubKey.Trim()
    # ACL: chi SYSTEM + Administrators
    icacls $authFile /inheritance:r | Out-Null
    icacls $authFile /grant 'SYSTEM:F' | Out-Null
    icacls $authFile /grant 'BUILTIN\Administrators:F' | Out-Null
    Write-Host "Public key -> $authFile (ACL admin OK)" -ForegroundColor Green
} else {
    $authFile = Join-Path $keyDir "authorized_keys"
    Add-Content -Path $authFile -Value $pubKey.Trim()
    icacls $authFile /inheritance:r | Out-Null
    icacls $authFile /grant 'NC:F' | Out-Null
    icacls $authFile /grant 'SYSTEM:F' | Out-Null
    Write-Host "Public key -> $authFile (ACL user OK)" -ForegroundColor Green
}

# --- 4. Restart sshd de chac chan ap dung ---
Restart-Service sshd
Write-Host "sshd restarted" -ForegroundColor Green

# --- 5. In private key de copy vao Termius ---
Write-Host ""
Write-Host "================ PRIVATE KEY (copy toan bo) ================" -ForegroundColor Yellow
Get-Content "$keyPath" -Raw
Write-Host "============================================================" -ForegroundColor Yellow
