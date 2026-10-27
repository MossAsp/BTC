"""
feature_importance.py -- วิเคราะห์ว่า feature ไหนสำคัญจริงกับโมเดลที่เทรนไว้แล้ว
ดึง feature_importances_ จากทั้ง 3 โมเดลย่อย (XGBoost, LightGBM, RandomForest)
แล้วเทียบกันเป็นตาราง + คะแนนเฉลี่ยรวม (normalize ให้แต่ละโมเดลรวมเป็น 1 ก่อนเฉลี่ย
เพราะแต่ละโมเดลมี scale ของ importance ไม่เหมือนกัน)

วิธีรัน (PowerShell):
    python feature_importance.py
หรือถ้าไม่ได้ activate venv ให้เรียก python เต็ม path จาก .venv โดยตรง

ผลลัพธ์: ตารางเปรียบเทียบ + บันทึกกราฟเป็น results/feature_importance.png
"""
from __future__ import annotations

import pandas as pd

from src.features import get_feature_columns
from src.model import EnsembleModel
from src.utils import PROJECT_ROOT, get_logger

logger = get_logger("feature_importance")

MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_model.joblib"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feature_matrix.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            "ไม่พบโมเดลที่เทรนไว้ — รัน `python -m main train` ก่อนอย่างน้อย 1 ครั้ง"
        )
    if not FEATURE_DATA_PATH.exists():
        raise SystemExit(
            "ไม่พบ feature_matrix.csv — รัน `python -m main fetch-data` ก่อน"
        )

    ensemble: EnsembleModel = EnsembleModel.load(MODEL_PATH)
    df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
    # ใช้รายชื่อฟีเจอร์ที่โมเดลจำไว้ตอนเทรน (ตรงกับลำดับที่โมเดลเห็นจริง) ถ้าไม่มี (โมเดลเก่า) ค่อยเดาจาก features.py
    feature_cols = getattr(ensemble, "feature_cols", None) or get_feature_columns(df)

    rows_per_model: dict[str, pd.Series] = {}
    for name, model in ensemble.models.items():
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"โมเดล {name} ไม่มี feature_importances_ ข้ามไป")
            continue
        importances = model.feature_importances_
        if len(importances) != len(feature_cols):
            logger.warning(
                f"จำนวน feature ของโมเดล {name} ({len(importances)}) ไม่ตรงกับ "
                f"feature_matrix.csv ({len(feature_cols)}) — อาจเป็นเพราะ retrain หลังแก้ features.py "
                f"แล้วยังไม่ได้ train ใหม่ ข้ามโมเดลนี้ไปก่อน"
            )
            continue
        series = pd.Series(importances, index=feature_cols, name=name)
        # normalize ให้รวมเป็น 1 เพื่อเทียบข้ามโมเดลได้ยุติธรรม (แต่ละโมเดลมี scale ต่างกัน)
        rows_per_model[name] = series / series.sum()

    if not rows_per_model:
        raise SystemExit(
            "ไม่สามารถดึง feature importance จากโมเดลใดได้เลย "
            "(น่าจะต้อง retrain โมเดลใหม่ให้ตรงกับ feature_matrix.csv ปัจจุบันก่อน)"
        )

    result_df = pd.DataFrame(rows_per_model)
    result_df["average"] = result_df.mean(axis=1)
    result_df = result_df.sort_values("average", ascending=False)

    print("\n=== Feature Importance เทียบระหว่างโมเดล (normalized, เรียงตามคะแนนเฉลี่ย) ===")
    print(result_df.round(4).to_string())

    print("\n=== Top 5 feature ที่สำคัญที่สุด (เฉลี่ยทุกโมเดล) ===")
    for feat, score in result_df["average"].head(5).items():
        print(f"  {feat}: {score:.4f}")

    print("\n=== Bottom 5 feature ที่สำคัญน้อยที่สุด (ผู้สมัครสำหรับตัดทิ้ง) ===")
    for feat, score in result_df["average"].tail(5).items():
        print(f"  {feat}: {score:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "feature_importance.csv"
    result_df.to_csv(csv_path)
    print(f"\nบันทึกผลลัพธ์ทั้งหมดไปที่ {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, max(6, len(result_df) * 0.35)))
        result_df["average"].sort_values().plot(kind="barh", ax=ax, color="#3498db")
        ax.set_xlabel("Normalized importance (mean of 3 models)")
        ax.set_title("Feature Importance — BTC/USDT Ensemble")
        plt.tight_layout()
        png_path = RESULTS_DIR / "feature_importance.png"
        plt.savefig(png_path, dpi=120)
        print(f"บันทึกกราฟไปที่ {png_path}")
    except ImportError:
        print("(ไม่มี matplotlib ติดตั้งอยู่ — ข้ามการพล็อตกราฟ ดูผลจาก CSV แทนได้)")


if __name__ == "__main__":
    main()
