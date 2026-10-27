# train_with_backup.ps1
# Use this instead of "python -m main train" directly.
# It backs up the existing model/data before training a new one, timestamped.
# Usage: place this file at project root (same folder as main.py), then run:
#   .\train_with_backup.ps1

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "backups\$timestamp"

Write-Host "=== Backing up existing model/data before training ===" -ForegroundColor Cyan

if ((Test-Path "models\ensemble_model.joblib") -or (Test-Path "data\oos_proba.csv")) {
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

    if (Test-Path "models\ensemble_model.joblib") {
        Copy-Item "models\ensemble_model.joblib" "$backupDir\ensemble_model.joblib"
        Write-Host "  backed up models\ensemble_model.joblib -> $backupDir\" -ForegroundColor Green
    }
    if (Test-Path "data\oos_proba.csv") {
        Copy-Item "data\oos_proba.csv" "$backupDir\oos_proba.csv"
        Write-Host "  backed up data\oos_proba.csv -> $backupDir\" -ForegroundColor Green
    }
    if (Test-Path "results\walk_forward_results.csv") {
        Copy-Item "results\walk_forward_results.csv" "$backupDir\walk_forward_results.csv"
        Write-Host "  backed up results\walk_forward_results.csv -> $backupDir\" -ForegroundColor Green
    }
    if (Test-Path "results\strategy_comparison.csv") {
        Copy-Item "results\strategy_comparison.csv" "$backupDir\strategy_comparison.csv"
        Write-Host "  backed up results\strategy_comparison.csv -> $backupDir\" -ForegroundColor Green
    }

    Write-Host "Backup complete: $backupDir" -ForegroundColor Cyan
} else {
    Write-Host "  No existing model found (this is the first training run). Skipping backup." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Starting training ===" -ForegroundColor Cyan
python -m main train

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "If the new result is worse than before, restore the old one with:" -ForegroundColor Yellow
Write-Host "  .\restore_backup.ps1 -Timestamp $timestamp" -ForegroundColor Yellow
