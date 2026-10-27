# ============================================================
# run_pipeline.ps1 — activate venv + รัน pipeline ทุกขั้นตอนรวดเดียว
# วิธีรัน: .\run_pipeline.ps1
# ============================================================
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

if (-not (Test-Path "venv")) {
    Write-Host "ยังไม่มี venv — รัน .\setup.ps1 ก่อน" -ForegroundColor Red
    exit 1
}

& ".\venv\Scripts\Activate.ps1"
python -m main all   # fetch -> train -> backtest -> compare -> report
if ($LASTEXITCODE -ne 0) { Write-Host "pipeline ล้มเหลว (exit code $LASTEXITCODE) ดู log ด้านบน" -ForegroundColor Red; exit 1 }
Write-Host "เสร็จแล้ว — ตาราง/กราฟสำหรับใส่เล่มอยู่ที่ results\report\ (เริ่มอ่านที่ REPORT_SUMMARY.md)" -ForegroundColor Green
