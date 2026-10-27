"""
sweep_stop_loss.py — ลอง stop_loss_pct หลายค่า (กับ take_profit_pct/position_sizing_mode
ที่ดีที่สุดของ ML_Ensemble จากผล sweep ก่อนหน้า: TP=0.08, dynamic) เพื่อดูว่าลด Max Drawdown
ได้โดยไม่ทำให้ Sharpe แย่ลงไหม

วางสคริปต์นี้ในโฟลเดอร์ root ของโปรเจกต์ (เดียวกับ main.py) แล้วรัน:
    python sweep_stop_loss.py
"""
import pandas as pd

from src.backtester import run_backtest, signals_from_confidence
from src.evaluation import load_oos
from src.utils import PROJECT_ROOT, load_config

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feature_matrix.csv"
OOS_PROBA_PATH = PROJECT_ROOT / "data" / "oos_proba.csv"

STOP_LOSS_VALUES = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
FIXED_TAKE_PROFIT = 0.08          # ค่าที่ดีที่สุดของ ML จาก sweep_tp_sizing.py
FIXED_SIZING_MODE = "dynamic"     # ค่าที่ดีที่สุดของ ML จาก sweep_tp_sizing.py

config = load_config()
df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
df_oos, proba_up = load_oos(df, OOS_PROBA_PATH)  # จับคู่ด้วยวันที่ ไม่พังแม้ fetch-data ทำให้เมทริกซ์ยาวขึ้น

base_cfg = dict(config["backtest"])
signal = signals_from_confidence(proba_up, base_cfg)

rows = []
for sl in STOP_LOSS_VALUES:
    cfg = {
        **base_cfg,
        "stop_loss_pct": sl,
        "take_profit_pct": FIXED_TAKE_PROFIT,
        "position_sizing_mode": FIXED_SIZING_MODE,
    }
    res = run_backtest(df_oos, signal, cfg, confidence=pd.Series(proba_up))["summary"]
    rows.append({
        "stop_loss_pct": sl,
        "n_trades": res["n_trades"],
        "total_return_pct": round(res["total_return_pct"], 2),
        "win_rate_pct": round(res["win_rate_pct"], 2),
        "sharpe_ratio": round(res["sharpe_ratio"], 3),
        "max_drawdown_pct": round(res["max_drawdown_pct"], 2),
    })

result_df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
print(f"(take_profit_pct={FIXED_TAKE_PROFIT}, position_sizing_mode={FIXED_SIZING_MODE} คงที่ทุกแถว)\n")
print(result_df.to_string(index=False))

best_sharpe = result_df.iloc[0]
best_dd = result_df.sort_values("max_drawdown_pct", ascending=False).iloc[0]
print(
    f"\n>>> Sharpe ดีที่สุด: stop_loss_pct={best_sharpe['stop_loss_pct']} "
    f"(Sharpe {best_sharpe['sharpe_ratio']}, Max DD {best_sharpe['max_drawdown_pct']}%)"
)
print(
    f">>> Max Drawdown ต่ำสุด (เสี่ยงน้อยสุด): stop_loss_pct={best_dd['stop_loss_pct']} "
    f"(Max DD {best_dd['max_drawdown_pct']}%, Sharpe {best_dd['sharpe_ratio']})"
)

out_path = PROJECT_ROOT / config["paths"]["results_dir"] / "stop_loss_sweep.csv"
result_df.to_csv(out_path, index=False)
print(f"\nบันทึกผลเรียบร้อย เขียนที่ {out_path}")
