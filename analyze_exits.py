"""
analyze_exits.py — สรุปว่าไม้ที่ขาดทุน/กำไรของ ML Ensemble ออกด้วยเหตุผลอะไรบ้าง
ใช้ก่อนวางสคริปต์นี้ในโฟลเดอร์ root ของโปรเจกต์ (เดียวกับ main.py) แล้วรันหลัง
`python -m main backtest` เสร็จ (ต้องรันคำสั่งนั้นก่อน เพื่อให้มี results/ml_trade_log.csv)

รัน: python analyze_exits.py
"""
import pandas as pd

from src.utils import PROJECT_ROOT, load_config

config = load_config()
log_path = PROJECT_ROOT / config["paths"]["results_dir"] / "ml_trade_log.csv"

if not log_path.exists():
    raise SystemExit(
        f"ไม่พบ {log_path} — รัน `python -m main backtest` ก่อน เพื่อให้สร้างไฟล์นี้"
    )

trades = pd.read_csv(log_path)
exits = trades[trades["exit_price"].notna()].copy()

if exits.empty:
    raise SystemExit("ไม่มีไม้ที่ปิดสถานะเลยในไฟล์นี้")

print(f"จำนวนไม้ทั้งหมดที่ปิดแล้ว: {len(exits)}\n")

# 1) สรุปจำนวนไม้ และ PnL เฉลี่ย แยกตามเหตุผลที่ออก
summary = exits.groupby("reason").agg(
    n_trades=("pnl", "count"),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
    avg_pnl_pct=("pnl_pct", "mean"),
    worst_pnl=("pnl", "min"),
    best_pnl=("pnl", "max"),
).round(2)
summary["pct_of_total_loss"] = (
    summary["total_pnl"].clip(upper=0) / exits.loc[exits["pnl"] < 0, "pnl"].sum() * 100
).round(1)
print("=== สรุปตาม exit reason ===")
print(summary.to_string())
print()

# 2) ไม้ที่ขาดทุนหนักสุด 10 อันดับแรก — ดูว่าเป็น outlier ไม่กี่ตัว หรือขาดทุนสม่ำเสมอ
worst = exits.nsmallest(10, "pnl")[["entry_date", "exit_date", "reason", "pnl", "pnl_pct"]]
print("=== 10 ไม้ที่ขาดทุนหนักที่สุด ===")
print(worst.to_string(index=False))
print()

# 3) เช็คว่าขาดทุนหนักกระจุกอยู่ไม่กี่ไม้ (outlier) หรือสม่ำเสมอทุกไม้
losses = exits.loc[exits["pnl"] < 0, "pnl"]
if len(losses) > 0:
    total_loss = losses.sum()
    top5_loss = losses.nsmallest(5).sum()
    print(f"ไม้ขาดทุนทั้งหมด {len(losses)} ไม้ รวม {total_loss:,.2f}")
    print(f"5 ไม้ที่ขาดทุนหนักสุด รวมกันคิดเป็น {top5_loss / total_loss * 100:.1f}% ของขาดทุนทั้งหมด")
    print(
        "-> ถ้าเปอร์เซ็นต์นี้สูง (เช่น >40%) แปลว่าขาดทุนหนักมาจาก outlier ไม่กี่ไม้ "
        "ควรดูว่าไม้พวกนี้เข้าช่วงราคาผันผวนสูงหรือไม่ (ลองเทียบกับ atr_pct ของวันนั้น)\n"
        "-> ถ้าเปอร์เซ็นต์นี้ต่ำ แปลว่าขาดทุนหนักสม่ำเสมอทุกไม้ที่โดน stop loss "
        "ปัญหาน่าจะอยู่ที่ stop_loss_pct กว้างเกินไปในภาพรวม ไม่ใช่แค่บางไม้"
    )
