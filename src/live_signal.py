"""
live_signal.py -- คำนวณสัญญาณเทรดล่าสุด (BUY/HOLD/SELL) จากข้อมูลปัจจุบันและโมเดลที่เทรนไว้
เป็นโมดูลกลางที่ discord_alert.py และ app.py (dashboard) ใช้ร่วมกัน เพื่อให้ตรรกะตรงกันเป๊ะ

จุดที่แก้จากเวอร์ชันก่อน:
  - ใช้ live_features.csv (ยาวถึงแท่งล่าสุด) ไม่ใช่ feature_matrix.csv (ถูกตัดท้าย target_horizon แถว)
    เดิมสัญญาณ/ราคา/ความผันผวนที่แสดงจึงเป็นของ 7 วันก่อน
  - ตัดสิน BUY/SELL/HOLD ด้วย signals_from_confidence() ตัวเดียวกับ backtest (รวม min_confidence_gap)
  - ใช้รายชื่อฟีเจอร์ที่ "โมเดลจำไว้ตอนเทรน" (model.feature_cols) กัน feature ไม่ตรงกันเงียบๆ
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import features
from src.backtester import signals_from_confidence
from src.model import EnsembleModel
from src.paths import FEATURE_DATA_PATH, LIVE_FEATURE_PATH
from src.utils import get_logger

logger = get_logger("live_signal")

STALE_AFTER_DAYS = 2  # แท่งรายวันล่าสุดที่ปิดแล้วควรเป็นของเมื่อวาน (UTC) ถ้าเก่ากว่านี้ถือว่าข้อมูลค้าง


def load_live_frame() -> pd.DataFrame:
    """โหลดเมทริกซ์สำหรับสัญญาณล่าสุด ถ้ายังไม่มี live_features.csv ให้ถอยไปใช้ feature_matrix.csv พร้อมเตือน"""
    if LIVE_FEATURE_PATH.exists():
        return pd.read_csv(LIVE_FEATURE_PATH, index_col="date", parse_dates=True)
    if FEATURE_DATA_PATH.exists():
        logger.warning(
            "ไม่พบ live_features.csv — ใช้ feature_matrix.csv แทน (สัญญาณจะเก่ากว่าปัจจุบัน target_horizon วัน) "
            "รัน `python -m main fetch-data` เพื่อสร้างไฟล์ที่ถูกต้อง"
        )
        return pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
    raise FileNotFoundError("ไม่พบ live_features.csv / feature_matrix.csv — รัน `python -m main fetch-data` ก่อน")


def classify_risk_level(volatility_7d_pct: float) -> tuple[str, str]:
    """
    แบ่งระดับความผันผวนเป็น LOW / MEDIUM / HIGH ตาม % การเปลี่ยนแปลงสะสม 7 วันล่าสุด
    เกณฑ์เป็นค่าประมาณคร่าวๆ สำหรับ BTC (ผันผวนสูงกว่าหุ้นทั่วไปเป็นปกติ) ปรับได้ตามความเหมาะสม
    """
    abs_vol = abs(volatility_7d_pct)
    if abs_vol < 3.0:
        return "LOW", "🟢"
    if abs_vol < 7.0:
        return "MEDIUM", "🟡"
    return "HIGH", "🔴"


_LABELS = {1: ("BUY", "เข้าซื้อ"), -1: ("SELL", "ระวังขาลง"), 0: ("HOLD", "ยังไม่ชัดเจน")}


def get_latest_signal(df: pd.DataFrame, model: EnsembleModel, config: dict) -> dict:
    """คำนวณสัญญาณจากแถวล่าสุดที่ฟีเจอร์ครบ คืน dict รวมทุกอย่างที่ต้องใช้แสดงผล"""
    backtest_cfg = config["backtest"]
    horizon = config["model"]["target_horizon"]

    feature_cols = getattr(model, "feature_cols", None) or features.get_feature_columns(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"โมเดลที่บันทึกไว้ใช้ฟีเจอร์ที่ไม่มีในข้อมูลปัจจุบัน (เช่น {missing[:3]}) "
            "— แก้ features.py หลังเทรนครั้งล่าสุด? รัน `python -m main train` ใหม่"
        )

    valid = df.dropna(subset=feature_cols)
    if valid.empty:
        raise ValueError("ไม่มีแถวที่ฟีเจอร์ครบเลย — รัน `python -m main fetch-data`")

    latest = valid.iloc[-1]
    proba_up = float(model.predict_proba(valid[feature_cols].iloc[[-1]])[0, 1])
    current_price = float(latest["btc_close"])

    # ความผันผวนสะสม 7 วันล่าสุด (% การเปลี่ยนแปลงราคาสุทธิ ไม่ใช่ std)
    if len(valid) >= 8:
        volatility_7d_pct = (current_price / float(valid["btc_close"].iloc[-8]) - 1) * 100
    else:
        volatility_7d_pct = 0.0
    risk_level, risk_emoji = classify_risk_level(volatility_7d_pct)

    # ตัดสินสัญญาณด้วยฟังก์ชันเดียวกับ backtest (threshold + min_confidence_gap)
    code = int(signals_from_confidence(np.array([proba_up]), backtest_cfg).iloc[0])
    signal_label, signal_text_th = _LABELS[code]

    as_of = valid.index[-1]
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    age_days = int((today_utc - pd.Timestamp(as_of).normalize()).days)

    return {
        "proba_up": proba_up,
        "confidence_pct": proba_up * 100,
        "current_price": current_price,
        "volatility_7d_pct": volatility_7d_pct,
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "signal_label": signal_label,       # "BUY" / "HOLD" / "SELL"
        "signal_text_th": signal_text_th,
        "threshold": backtest_cfg["confidence_threshold"],
        "horizon": horizon,
        "as_of_date": as_of,
        "data_age_days": age_days,
        "is_stale": age_days > STALE_AFTER_DAYS,
        "return_1d_pct": float(latest["return_1d"]) * 100 if "return_1d" in latest else float("nan"),
        "rsi": float(latest["rsi"]) if "rsi" in latest else float("nan"),
        "regime": int(latest["regime"]) if "regime" in latest else 0,
    }
