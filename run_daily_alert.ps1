# run_daily_alert.ps1 -- สคริปต์สำหรับ Windows Task Scheduler
# ดึงข้อมูลล่าสุด + ส่งแจ้งเตือน Discord วันละครั้ง
# ตั้งเวลาให้รันหลังแท่งรายวันปิด: Binance ปิดแท่ง 00:00 UTC = 07:00 เวลาไทย -> แนะนำ 07:30 ขึ้นไป
#
# ตั้งใน Task Scheduler:  Program: powershell.exe
#   Arguments: -ExecutionPolicy Bypass -File "C:\path\to\btc_pipeline\run_daily_alert.ps1"
# ผลการรันทุกครั้งบันทึกที่ logs\daily_alert.log (ดูย้อนหลังได้ว่าวันไหนล้มเหลว)

Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Path "logs" -Force | Out-Null
$logFile = "logs\daily_alert.log"

function Write-Log($message) {
    $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message
    Add-Content -Path $logFile -Value $line -Encoding UTF8
    Write-Host $line
}

& ".\venv\Scripts\Activate.ps1"

Write-Log "เริ่ม: ดึงข้อมูลใหม่ (force refresh)"
python -m main fetch-data --force-refresh
if ($LASTEXITCODE -ne 0) {
    Write-Log "ล้มเหลว: fetch-data (exit code $LASTEXITCODE) -- ไม่ส่ง alert เพื่อกันส่งสัญญาณจากข้อมูลเก่า (ดูสาเหตุใน logs\data_loader.log)"
    exit 1
}

Write-Log "ส่ง Discord alert"
python discord_alert.py
if ($LASTEXITCODE -ne 0) {
    Write-Log "ล้มเหลว: discord_alert (exit code $LASTEXITCODE)"
    exit 1
}

Write-Log "เสร็จสิ้น"
