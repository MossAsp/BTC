"""
sweep_tp_sizing.py — ลอง take_profit_pct หลายค่า x position_sizing_mode (fixed/dynamic)
บนข้อมูล OOS เดียวกัน เพื่อหาคอมโบที่ Sharpe ดีที่สุด หลังแก้บั๊ก stop loss/take profit
(เช็คจาก low/high แทน close) ไปแล้ว

วางสคริปต์นี้ในโฟลเดอร์ root ของโปรเจกต์ (เดียวกับ main.py) แล้วรัน:
    python sweep_tp_sizing.py

หมายเหตุ: config/plan.yaml เคยมีคอมเมนต์ว่า dynamic sizing แย่กว่า fixed เสมอ
(Sharpe 0.306 vs 0.214) แต่นั่นเป็นผลทดสอบ "ก่อน" แก้บั๊ก stop loss สคริปต์นี้ทดสอบซ้ำ
บน backtester เวอร์ชันที่แก้บั๊กแล้ว เพื่อยืนยันว่าข้อสรุปเดิมยังจริงอยู่ไหม
"""
import pandas as pd

from src.backtester import run_backtest, signals_from_confidence
from src.evaluation import load_oos
from src.utils import PROJECT_ROOT, load_config

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feature_matrix.csv"
OOS_PROBA_PATH = PROJECT_ROOT / "data" / "oos_proba.csv"

TAKE_PROFIT_VALUES = [0.08, 0.10, 0.12, 0.15]
SIZING_MODES = ["fixed", "dynamic"]

config = load_config()
df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
df_oos, proba_up = load_oos(df, OOS_PROBA_PATH)  # จับคู่ด้วยวันที่ ไม่พังแม้ fetch-data ทำให้เมทริกซ์ยาวขึ้น

base_cfg = dict(config["backtest"])
signal = signals_from_confidence(proba_up, base_cfg)

rows = []
for tp in TAKE_PROFIT_VALUES:
    for mode in SIZING_MODES:
        cfg = {**base_cfg, "take_profit_pct": tp, "position_sizing_mode": mode}
        res = run_backtest(df_oos, signal, cfg, confidence=pd.Series(proba_up))["summary"]
        rows.append({
            "take_profit_pct": tp,
            "sizing_mode": mode,
            "n_trades": res["n_trades"],
            "total_return_pct": round(res["total_return_pct"], 2),
            "win_rate_pct": round(res["win_rate_pct"], 2),
            "sharpe_ratio": round(res["sharpe_ratio"], 3),
            "max_drawdown_pct": round(res["max_drawdown_pct"], 2),
        })

result_df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
print(result_df.to_string(index=False))

best = result_df.iloc[0]
print(
    f"\n>>> คอมโบ Sharpe ดีที่สุด: take_profit_pct={best['take_profit_pct']}, "
    f"sizing_mode={best['sizing_mode']} (Sharpe {best['sharpe_ratio']}, "
    f"Return {best['total_return_pct']}%, Max DD {best['max_drawdown_pct']}%)"
)

out_path = PROJECT_ROOT / config["paths"]["results_dir"] / "tp_sizing_sweep.csv"
result_df.to_csv(out_path, index=False)
print(f"บันทึกผลเรียบร้อย เขียนที่ {out_path}")
