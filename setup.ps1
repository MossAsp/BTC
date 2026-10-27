# ============================================================
# setup.ps1 — ติดตั้งทุกอย่างให้อัตโนมัติ รันครั้งเดียวจบ
# วิธีรัน: เปิด PowerShell ที่โฟลเดอร์ btc_pipeline แล้วพิมพ์
#   .\setup.ps1
# ============================================================

Write-Host "=== 1. อนุญาตให้รันสคริปต์ใน session นี้ (ไม่กระทบเครื่องถาวร) ===" -ForegroundColor Cyan
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

Write-Host "=== 2. สร้าง virtual environment (venv) ===" -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "พบ venv เดิมอยู่แล้ว ข้ามขั้นตอนสร้างใหม่ (ถ้าอยากสร้างใหม่ ให้ลบโฟลเดอร์ venv ก่อน)" -ForegroundColor Yellow
} else {
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "สร้าง venv ไม่สำเร็จ — เช็คว่าติดตั้ง Python แล้วหรือยัง (python --version)" -ForegroundColor Red
        exit 1
    }
}

Write-Host "=== 3. Activate venv ===" -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

Write-Host "=== 4. อัปเกรด pip ===" -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

Write-Host "=== 5. ติดตั้ง dependencies (เวอร์ชันถูกล็อกไว้แล้ว กันปัญหา DLL ถูกบล็อก) ===" -ForegroundColor Cyan
python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ติดตั้ง dependencies ไม่สำเร็จ — copy ข้อความ error ด้านบนส่งไปให้ Claude ช่วยดูได้เลย" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== เสร็จแล้ว! ===" -ForegroundColor Green
Write-Host "ต่อไปรันคำสั่งเหล่านี้ได้เลย (venv ยัง active อยู่ในหน้าต่างนี้):" -ForegroundColor Green
Write-Host "  python -m main fetch-data"
Write-Host "  python -m main train"
Write-Host "  python -m main backtest"
Write-Host "  python -m main compare"
Write-Host "  python -m main report      # ตาราง/กราฟสำหรับใส่เล่ม -> results\report\"
Write-Host "  python -m streamlit run app.py"
Write-Host ""
Write-Host "ครั้งต่อไปที่เปิด terminal ใหม่ ต้อง activate venv ก่อนเสมอด้วยคำสั่ง:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1"
