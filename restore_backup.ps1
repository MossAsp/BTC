# restore_backup.ps1
# Restore model/data from a backup created by train_with_backup.ps1
#
# Usage:
#   List all backups:              .\restore_backup.ps1
#   Restore a specific backup:     .\restore_backup.ps1 -Timestamp 20260904_153000

param(
    [string]$Timestamp = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "backups")) {
    Write-Host "No backups folder found yet (you have not trained via train_with_backup.ps1 yet)" -ForegroundColor Yellow
    exit 1
}

if ($Timestamp -eq "") {
    Write-Host "=== Available backups (newest first) ===" -ForegroundColor Cyan
    Get-ChildItem "backups" -Directory | Sort-Object Name -Descending | ForEach-Object {
        Write-Host "  $($_.Name)"
    }
    Write-Host ""
    Write-Host "Usage: .\restore_backup.ps1 -Timestamp <backup folder name>" -ForegroundColor Yellow
    exit 0
}

$backupDir = "backups\$Timestamp"

if (-not (Test-Path $backupDir)) {
    Write-Host "Backup '$Timestamp' not found. Run .\restore_backup.ps1 with no arguments to see the list." -ForegroundColor Red
    exit 1
}

Write-Host "=== Restoring from $backupDir ===" -ForegroundColor Cyan

if (Test-Path "$backupDir\ensemble_model.joblib") {
    New-Item -ItemType Directory -Path "models" -Force | Out-Null
    Copy-Item "$backupDir\ensemble_model.joblib" "models\ensemble_model.joblib" -Force
    Write-Host "  restored models\ensemble_model.joblib" -ForegroundColor Green
}
if (Test-Path "$backupDir\oos_proba.csv") {
    New-Item -ItemType Directory -Path "data" -Force | Out-Null
    Copy-Item "$backupDir\oos_proba.csv" "data\oos_proba.csv" -Force
    Write-Host "  restored data\oos_proba.csv" -ForegroundColor Green
}
if (Test-Path "$backupDir\walk_forward_results.csv") {
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
    Copy-Item "$backupDir\walk_forward_results.csv" "results\walk_forward_results.csv" -Force
    Write-Host "  restored results\walk_forward_results.csv" -ForegroundColor Green
}
if (Test-Path "$backupDir\strategy_comparison.csv") {
    New-Item -ItemType Directory -Path "results" -Force | Out-Null
    Copy-Item "$backupDir\strategy_comparison.csv" "results\strategy_comparison.csv" -Force
    Write-Host "  restored results\strategy_comparison.csv" -ForegroundColor Green
}

Write-Host ""
Write-Host "Restore complete. Run backtest/compare again to confirm:" -ForegroundColor Cyan
Write-Host "  python -m main backtest" -ForegroundColor Yellow
Write-Host "  python -m main compare" -ForegroundColor Yellow
