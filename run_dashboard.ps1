# ============================================================
# run_dashboard.ps1 — activate venv + เปิด Streamlit dashboard
# วิธีรัน: .\run_dashboard.ps1
# ============================================================
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

if (-not (Test-Path "venv")) {
    Write-Host "ยังไม่มี venv — รัน .\setup.ps1 ก่อน" -ForegroundColor Red
    exit 1
}

& ".\venv\Scripts\Activate.ps1"
python -m streamlit run app.py
