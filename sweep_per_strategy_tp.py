"""
sweep_per_strategy_tp.py — หา take_profit_pct ที่ดีที่สุดของแต่ละกลยุทธ์แยกกัน
แล้วเทียบ ML_Ensemble กับ rule-based strategies แบบ fair (ต่างคนต่าง optimize TP ของตัวเอง)
ไม่ใช้ TP เดียวร่วมกันแบบที่ sweep_tp_sizing.py เคยทำ (ซึ่งทำให้ TP ถูก optimize ให้ ML
โดยเฉพาะ แล้วไปกระทบผลของ MACD/RSI/Volatility ด้วย)

วางสคริปต์นี้ในโฟลเดอร์ root ของโปรเจกต์ (เดียวกับ main.py) แล้วรัน:
    python sweep_per_strategy_tp.py
"""
import pandas as pd

from src.backtester import run_backtest, signals_from_confidence
from src.rule_strategies import get_all_strategies
from src.evaluation import load_oos
from src.utils import PROJECT_ROOT, load_config

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feature_matrix.csv"
OOS_PROBA_PATH = PROJECT_ROOT / "data" / "oos_proba.csv"
TAKE_PROFIT_VALUES = [0.06, 0.08, 0.10, 0.12, 0.15]
SIZING_MODES_FOR_ML = ["fixed", "dynamic"]  # เฉพาะ ML ที่มี confidence ไว้คำนวณ dynamic sizing

config = load_config()
df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
df_oos, proba_up = load_oos(df, OOS_PROBA_PATH)  # จับคู่ด้วยวันที่ ไม่พังแม้ fetch-data ทำให้เมทริกซ์ยาวขึ้น

base_cfg = dict(config["backtest"])
ml_signal = signals_from_confidence(proba_up, base_cfg)
rule_signals = get_all_strategies(df_oos)

all_rows = []

# --- rule-based strategies: sweep TP, fixed position sizing (ไม่มี confidence ให้ใช้) ---
for name, sig in rule_signals.items():
    for tp in TAKE_PROFIT_VALUES:
        cfg = {**base_cfg, "take_profit_pct": tp, "position_sizing_mode": "fixed"}
        res = run_backtest(df_oos, sig, cfg)["summary"]
        all_rows.append({"strategy": name, "take_profit_pct": tp, "sizing_mode": "fixed", **res})

# --- ML_Ensemble: sweep TP x sizing mode ---
for tp in TAKE_PROFIT_VALUES:
    for mode in SIZING_MODES_FOR_ML:
        cfg = {**base_cfg, "take_profit_pct": tp, "position_sizing_mode": mode}
        res = run_backtest(df_oos, ml_signal, cfg, confidence=pd.Series(proba_up))["summary"]
        all_rows.append({"strategy": "ML_Ensemble", "take_profit_pct": tp, "sizing_mode": mode, **res})

full_df = pd.DataFrame(all_rows)
out_dir = PROJECT_ROOT / config["paths"]["results_dir"]
full_df.to_csv(out_dir / "per_strategy_tp_sweep_full.csv", index=False)

# --- หา TP (และ sizing mode สำหรับ ML) ที่ดีที่สุดของแต่ละกลยุทธ์ (ตาม Sharpe) ---
best_rows = []
for name, group in full_df.groupby("strategy"):
    best = group.loc[group["sharpe_ratio"].idxmax()]
    best_rows.append(best)

best_df = pd.DataFrame(best_rows).sort_values("sharpe_ratio", ascending=False)
cols = ["strategy", "take_profit_pct", "sizing_mode", "n_trades", "total_return_pct",
        "win_rate_pct", "sharpe_ratio", "max_drawdown_pct"]
print("=== แต่ละกลยุทธ์ที่ TP ของตัวเอง (fair comparison) ===")
print(best_df[cols].round(3).to_string(index=False))

best_df[cols].to_csv(out_dir / "per_strategy_tp_sweep_best.csv", index=False)
print(f"\nบันทึกผลละเอียดที่ {out_dir / 'per_strategy_tp_sweep_full.csv'}")
print(f"บันทึกผลสรุป (TP ดีที่สุดต่อกลยุทธ์) ที่ {out_dir / 'per_strategy_tp_sweep_best.csv'}")

winner = best_df.iloc[0]
if winner["strategy"] == "ML_Ensemble":
    print(f"\n>>> ML_Ensemble ยังเป็นอันดับ 1 ด้าน Sharpe แม้เทียบแบบ fair (TP ของใครของมัน)")
else:
    print(f"\n>>> {winner['strategy']} แซง ML_Ensemble เมื่อเทียบแบบ fair (TP ของใครของมัน)")
