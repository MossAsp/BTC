"""
sweep_threshold.py -- ลองไล่ confidence_threshold หลายค่าบนชุดข้อมูลเดียวกัน
เพื่อหาว่าค่าไหนให้ผล backtest ดีที่สุด (ไม่ต้อง retrain โมเดลใหม่ ใช้ oos_proba.csv เดิม)

วิธีรัน (PowerShell):
    python sweep_threshold.py
หรือถ้าไม่ได้ activate venv ให้เรียก python เต็ม path จาก .venv โดยตรง

ผลลัพธ์: ตารางเปรียบเทียบทุก threshold + บันทึกกราฟเป็น results/threshold_sweep.png
"""
from __future__ import annotations

import pandas as pd

from src.backtester import run_backtest, signals_from_confidence
from src.evaluation import load_oos
from src.utils import PROJECT_ROOT, load_config, get_logger

logger = get_logger("sweep_threshold")

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feature_matrix.csv"
OOS_PROBA_PATH = PROJECT_ROOT / "data" / "oos_proba.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    if not FEATURE_DATA_PATH.exists() or not OOS_PROBA_PATH.exists():
        raise SystemExit(
            "ไม่พบ feature_matrix.csv หรือ oos_proba.csv — รัน `python -m main fetch-data` "
            "และ `python -m main train` ก่อนอย่างน้อย 1 ครั้ง"
        )

    config = load_config()
    df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
    df_oos, proba_up = load_oos(df, OOS_PROBA_PATH)  # จับคู่ด้วยวันที่

    # ไล่ threshold ตั้งแต่ 0.50 ถึง 0.95 ทีละ 0.01
    thresholds = [round(0.50 + i * 0.01, 2) for i in range(46)]

    rows = []
    for t in thresholds:
        cfg = dict(config["backtest"])
        cfg["confidence_threshold"] = t

        signal = signals_from_confidence(proba_up, cfg)
        result = run_backtest(df_oos, signal, cfg, confidence=pd.Series(proba_up))
        summary = result["summary"]

        rows.append({
            "threshold": t,
            "n_trades": summary["n_trades"],
            "total_return_pct": round(summary["total_return_pct"], 2),
            "win_rate_pct": round(summary["win_rate_pct"], 2),
            "max_drawdown_pct": round(summary["max_drawdown_pct"], 2),
            "sharpe_ratio": round(summary["sharpe_ratio"], 3),
            "final_equity": round(summary["final_equity"], 2),
        })
        logger.info(f"threshold={t:.2f} -> {rows[-1]}")

    result_df = pd.DataFrame(rows)

    # เรียงตาม Sharpe Ratio จากดีสุดไปแย่สุด (ตัวชี้วัดที่สมดุลระหว่าง return กับความเสี่ยง)
    result_df_sorted = result_df.sort_values("sharpe_ratio", ascending=False)

    print("\n=== ผลลัพธ์ทุก threshold (เรียงตาม Sharpe Ratio ดีที่สุดก่อน) ===")
    print(result_df_sorted.to_string(index=False))

    best = result_df_sorted.iloc[0]
    print(f"\n=== threshold ที่ดีที่สุด (ตาม Sharpe Ratio) ===")
    print(f"threshold = {best['threshold']}")
    print(f"  n_trades = {best['n_trades']}")
    print(f"  total_return_pct = {best['total_return_pct']}%")
    print(f"  sharpe_ratio = {best['sharpe_ratio']}")
    print(f"  max_drawdown_pct = {best['max_drawdown_pct']}%")

    # บันทึกผลเป็น CSV ไว้ดูย้อนหลังได้
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "threshold_sweep.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\nบันทึกผลลัพธ์ทั้งหมดไปที่ {csv_path}")

    # พล็อตกราฟเปรียบเทียบ (ถ้ามี matplotlib ติดตั้งอยู่)
    try:
        import matplotlib
        matplotlib.use("Agg")  # ไม่ต้องเปิดหน้าต่าง GUI ป้องกัน error บน headless environment
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(result_df["threshold"], result_df["total_return_pct"], marker="o", color="#2ecc71")
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Total Return (%)")
        axes[0].set_title("Confidence Threshold Sweep")

        axes[1].plot(result_df["threshold"], result_df["sharpe_ratio"], marker="o", color="#3498db")
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Sharpe Ratio")

        axes[2].plot(result_df["threshold"], result_df["n_trades"], marker="o", color="#e67e22")
        axes[2].set_ylabel("Number of Trades")
        axes[2].set_xlabel("Confidence Threshold")

        plt.tight_layout()
        png_path = RESULTS_DIR / "threshold_sweep.png"
        plt.savefig(png_path, dpi=120)
        print(f"บันทึกกราฟไปที่ {png_path}")
    except ImportError:
        print("(ไม่มี matplotlib ติดตั้งอยู่ — ข้ามการพล็อตกราฟ ดูผลจาก CSV แทนได้)")


if __name__ == "__main__":
    main()
