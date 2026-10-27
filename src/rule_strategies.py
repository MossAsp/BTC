"""
rule_strategies.py — กลยุทธ์ rule-based ใช้เป็น baseline เทียบกับโมเดล ML
สำคัญ: baseline พวกนี้ช่วยวินิจฉัยว่าโมเดล ML underperform เพราะ feature ไม่พอ
หรือเพราะ threshold ผิด — ถ้า rule-based ก็แพ้ตลาดเหมือนกัน แปลว่าปัญหาไม่ใช่ที่โมเดล
"""
from __future__ import annotations

import pandas as pd

SIGNAL_BUY = 1
SIGNAL_SELL = -1
SIGNAL_HOLD = 0


def rsi_reversal(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """
    ซื้อเมื่อ RSI ต่ำกว่า oversold (ราคาลงมากเกินไป, คาดว่าจะเด้ง)
    ขายเมื่อ RSI สูงกว่า overbought (ราคาขึ้นมากเกินไป, คาดว่าจะย่อ)
    """
    signal = pd.Series(SIGNAL_HOLD, index=df.index)
    signal[df["rsi"] < oversold] = SIGNAL_BUY
    signal[df["rsi"] > overbought] = SIGNAL_SELL
    return signal


def macd_crossover(df: pd.DataFrame) -> pd.Series:
    """
    ซื้อเมื่อ MACD ตัดขึ้นเหนือ signal line (momentum เปลี่ยนเป็นขาขึ้น)
    ขายเมื่อ MACD ตัดลงใต้ signal line
    """
    signal = pd.Series(SIGNAL_HOLD, index=df.index)
    macd_diff = df["macd"] - df["macd_signal"]
    prev_diff = macd_diff.shift(1)

    cross_up = (prev_diff <= 0) & (macd_diff > 0)
    cross_down = (prev_diff >= 0) & (macd_diff < 0)

    signal[cross_up] = SIGNAL_BUY
    signal[cross_down] = SIGNAL_SELL
    return signal


def volatility_breakout(df: pd.DataFrame, atr_multiplier: float = 1.5) -> pd.Series:
    """
    เวอร์ชัน crypto ของ 'VIX Panic Reversal' เดิม (crypto ไม่มี VIX ของตัวเอง จึงใช้ ATR แทน)
    ซื้อเมื่อราคาทะลุกรอบขาขึ้นแรงกว่าปกติ (breakout พร้อม volume ยืนยัน)
    ขายเมื่อ volatility (VIX มหภาค) พุ่งสูงผิดปกติ ซึ่งมักกระทบ risk asset รวมถึง BTC
    """
    signal = pd.Series(SIGNAL_HOLD, index=df.index)

    breakout_up = (df["return_1d"] > df["atr_pct"] * atr_multiplier) & (df["volume_ma_ratio"] > 1.2)
    signal[breakout_up] = SIGNAL_BUY

    if "vix_spike" in df.columns:
        signal[df["vix_spike"] == 1] = SIGNAL_SELL

    return signal


def get_all_strategies(df: pd.DataFrame) -> dict[str, pd.Series]:
    """คืน dict ของสัญญาณจากทุก rule-based strategy เพื่อใช้เทียบ backtest"""
    return {
        "RSI_Reversal": rsi_reversal(df),
        "MACD_Crossover": macd_crossover(df),
        "Volatility_Breakout": volatility_breakout(df),
    }
