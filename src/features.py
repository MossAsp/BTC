"""
features.py — สร้างฟีเจอร์ทั้งหมดสำหรับโมเดล:
  1. Technical indicators (RSI, MACD, Bollinger Bands, ATR) — เขียนเองด้วย pandas/numpy ล้วน
  2. 3-class regime classification (bull / bear / sideways)
  3. Cross-asset signals (ETH, DXY, VIX) และ on-chain
  4. Target label (ทิศทางราคาล่วงหน้า N แท่ง)

มี "สองเมทริกซ์" ที่ใช้ต่างวัตถุประสงค์:
  - build_feature_matrix : สำหรับเทรน/backtest  -> ตัดแถวท้ายที่ยังไม่มี target ออก (ยาวไม่ถึงแท่งล่าสุด)
  - build_live_matrix    : สำหรับสัญญาณล่าสุด    -> ไม่ต้องมี target จึงยาวถึงแท่งที่ปิดล่าสุดจริง
    (เดิมสัญญาณ "ล่าสุด" ใช้แถวท้ายของเมทริกซ์แรก ซึ่งเก่ากว่าปัจจุบัน target_horizon วัน)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("features")

REGIME_BULL = 1
REGIME_SIDEWAYS = 0
REGIME_BEAR = -1

# ราคา/ระดับดิบ: ไม่ใช้เป็น feature เพราะ non-stationary (BTC 2022 ~20k vs 2025 >100k)
# ต้นไม้ตัดสินใจแบ่งตามระดับราคาไม่ได้ generalize ข้ามช่วงเวลา -> ใช้เวอร์ชัน normalize แทน
RAW_PRICE_COLS = {
    "btc_open", "btc_high", "btc_low", "btc_close", "btc_volume",
    "eth_close", "dxy_close", "vix_close",
    "onchain_active_addresses", "onchain_hash_rate",
}
PRICE_LEVEL_INDICATOR_COLS = {
    "bb_upper", "bb_mid", "bb_lower", "atr", "macd", "macd_signal", "macd_hist",
}
LABEL_COLS = {"target", "future_return", "regime"}
NON_FEATURE_COLS = RAW_PRICE_COLS | PRICE_LEVEL_INDICATOR_COLS | LABEL_COLS


# ============================================================
# Technical indicators เขียนเองด้วย pandas/numpy ล้วนๆ
# (ไม่ใช้ pandas-ta เพื่อลดจำนวน dependency ภายนอกที่อาจโดน
# Windows Application Control Policy บล็อก และคุมเวอร์ชัน pandas/numpy ได้เอง)
# ============================================================

def _rsi(close: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index มาตรฐาน (Wilder's smoothing)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # ไม่มีวันลงเลยในหน้าต่าง (avg_loss == 0) -> RSI = 100 (เดิมได้ NaN แล้วโดน dropna ตัดแถวทิ้ง)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger_bands(close: pd.Series, length: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, mid (SMA), lower"""
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    return mid + n_std * std, mid, mid - n_std * std


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range (Wilder's smoothing)"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def add_technical_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """เพิ่ม technical indicators จากราคา BTC (คอลัมน์ดิบเก็บไว้ให้ rule-based strategy ใช้ แต่ไม่เข้าโมเดล)"""
    out = df.copy()
    close, high, low, volume = out["btc_close"], out["btc_high"], out["btc_low"], out["btc_volume"]

    out["rsi"] = _rsi(close, cfg["rsi_period"])

    macd_line, macd_signal, macd_hist = _macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    # เวอร์ชัน normalize ด้วยราคา (stationary) — ตัวที่เข้าโมเดลจริง
    out["macd_pct"] = macd_line / close
    out["macd_signal_pct"] = macd_signal / close
    out["macd_hist_pct"] = macd_hist / close

    bb_upper, bb_mid, bb_lower = _bollinger_bands(close, cfg["bb_period"], cfg["bb_std"])
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower)
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid       # ความกว้างแบนด์ต่อราคา (volatility squeeze)
    out["close_to_bb_mid"] = close / bb_mid - 1            # ราคาห่างจากเส้นกลางกี่ %

    out["atr"] = _atr(high, low, close, cfg["atr_period"])
    out["atr_pct"] = out["atr"] / close  # normalize ด้วยราคา เพื่อเทียบข้ามช่วงเวลาได้

    out["volume_change"] = volume.pct_change()
    out["volume_ma_ratio"] = volume / volume.rolling(20).mean()

    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["volatility_20d"] = out["return_1d"].rolling(20).std()

    return out


def add_regime_classification(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    จำแนก regime เป็น 3 คลาส: bull (1) / sideways (0) / bear (-1)
    ใช้ % การเปลี่ยนแปลงของราคาในช่วง regime_window วัน (ย้อนหลังเท่านั้น ไม่มี look-ahead) เทียบกับ threshold

    หมายเหตุ: คอลัมน์ regime ใช้แสดงผลใน dashboard เท่านั้น ไม่เข้าโมเดล (ดู NON_FEATURE_COLS)
    เพราะ one-hot ของมันมี feature importance ใกล้ 0 ในทุกโมเดล (วิเคราะห์เมื่อ 18 ส.ค.)
    """
    out = df.copy()
    window = cfg["regime_window"]
    threshold = cfg["regime_threshold_pct"]

    trend_pct = out["btc_close"].pct_change(window)
    conditions = [trend_pct > threshold, trend_pct < -threshold]
    out["regime"] = np.select(conditions, [REGIME_BULL, REGIME_BEAR], default=REGIME_SIDEWAYS)
    return out


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ฟีเจอร์จากสินทรัพย์อื่น:
      - ETH: อัตราส่วน ETH/BTC, ผลตอบแทน 5 วัน, correlation 20 วันกับ BTC
      - DXY: ดัชนีดอลลาร์ (แข็ง -> risk-off มักกดดัน BTC)
      - VIX: ดัชนีความกลัวตลาดหุ้น (พุ่ง -> risk-off ทั่วตลาด)
      - on-chain: ใช้ % change / z-score แทนค่าดิบ เพราะค่าดิบโตตามเครือข่าย (non-stationary)
    """
    out = df.copy()

    if "eth_close" in out.columns:
        out["eth_btc_ratio"] = out["eth_close"] / out["btc_close"]
        out["eth_return_5d"] = out["eth_close"].pct_change(5)
        out["btc_eth_corr_20d"] = (
            out["btc_close"].pct_change().rolling(20).corr(out["eth_close"].pct_change())
        )

    if "dxy_close" in out.columns:
        out["dxy_return_5d"] = out["dxy_close"].pct_change(5)
        out["dxy_zscore_20d"] = (
            out["dxy_close"] - out["dxy_close"].rolling(20).mean()
        ) / out["dxy_close"].rolling(20).std()

    if "vix_close" in out.columns:
        out["vix_zscore_20d"] = (
            out["vix_close"] - out["vix_close"].rolling(20).mean()
        ) / out["vix_close"].rolling(20).std()
        out["vix_spike"] = (out["vix_close"] > out["vix_close"].rolling(20).mean() * 1.2).astype(int)

    if "onchain_active_addresses" in out.columns:
        out["onchain_active_addr_change_7d"] = out["onchain_active_addresses"].pct_change(7)
        out["onchain_active_addr_zscore_30d"] = (
            out["onchain_active_addresses"] - out["onchain_active_addresses"].rolling(30).mean()
        ) / out["onchain_active_addresses"].rolling(30).std()

    if "onchain_hash_rate" in out.columns:
        out["onchain_hash_rate_change_7d"] = out["onchain_hash_rate"].pct_change(7)

    return out


def add_target_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    target: ราคาจะขึ้น (1) หรือลง (0) ใน horizon แท่งข้างหน้า
    แถวท้าย horizon แถวยังไม่รู้อนาคต -> future_return และ target เป็น NaN (ไม่ใช่ 0 แบบเดิมที่ปนเข้ามาเป็น label ปลอม)
    """
    out = df.copy()
    future_return = out["btc_close"].shift(-horizon) / out["btc_close"] - 1
    out["future_return"] = future_return
    out["target"] = (future_return > 0).astype(float).where(future_return.notna())
    return out


def build_all_features(raw_merged: pd.DataFrame, config: dict) -> pd.DataFrame:
    """คำนวณฟีเจอร์ + target ทั้งหมดโดยยังไม่ตัดแถวใดทิ้ง (ใช้เป็นฐานของ 2 เมทริกซ์ด้านล่าง)"""
    feat_cfg = config["features"]
    df = add_technical_indicators(raw_merged, feat_cfg)
    if feat_cfg.get("regime_classification", True):
        df = add_regime_classification(df, feat_cfg)
    df = add_cross_asset_features(df)
    df = add_target_label(df, config["model"]["target_horizon"])
    return df.replace([np.inf, -np.inf], np.nan)


def build_feature_matrix(raw_merged: pd.DataFrame, config: dict) -> pd.DataFrame:
    """เมทริกซ์สำหรับเทรน/backtest: ตัดทุกแถวที่มี NaN (รวมแถวท้ายที่ยังไม่มี target)"""
    df = build_all_features(raw_merged, config)
    n_before = len(df)
    df = df.dropna()
    df["target"] = df["target"].astype(int)
    logger.info(
        f"สร้างเมทริกซ์เทรนเสร็จ: {len(df)} แถว (ตัดออก {n_before - len(df)} แถวที่มี NaN จาก rolling window "
        f"และแถวท้ายที่ยังไม่มี target)"
    )
    return df


def build_live_matrix(raw_merged: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    เมทริกซ์สำหรับสัญญาณล่าสุด: ตัดเฉพาะแถวที่ "ฟีเจอร์" เป็น NaN (ช่วงอุ่นเครื่องของ indicator)
    แถวท้ายที่ยังไม่มี target ยังอยู่ -> แถวสุดท้ายคือแท่งที่ปิดล่าสุดจริง
    """
    df = build_all_features(raw_merged, config)
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols)
    logger.info(f"สร้างเมทริกซ์สัญญาณล่าสุดเสร็จ: {len(df)} แถว ถึงวันที่ {df.index[-1].date()}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """คืนรายชื่อคอลัมน์ที่เป็น feature จริง (ตัดราคาดิบ, indicator ระดับราคา, target, regime ออก)"""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
