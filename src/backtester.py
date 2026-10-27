"""
backtester.py — Backtest engine เขียนด้วย pure pandas ล้วนๆ (ไม่ใช้ vectorbt
เพื่อเลี่ยงปัญหา DLL บน Windows ที่เจอมาก่อนหน้านี้)

รองรับทั้งสัญญาณจากโมเดล ML (พร้อม confidence threshold) และสัญญาณจาก rule-based strategy
สมมติฐานการ execute: สัญญาณคำนวณจากแท่งปิดวัน t แล้วเข้า/ออกที่ราคาปิดวัน t (+ slippage + fee)
ซึ่งใกล้เคียงความจริงของ BTC เพราะตลาด 24/7 ราคาเปิดวันถัดไป ≈ ราคาปิดวันก่อนหน้า
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("backtester")

TRADING_DAYS_PER_YEAR = 365  # BTC เทรดทุกวัน (ไม่ใช่ 252 แบบตลาดหุ้น)


def signals_from_confidence(proba_up: np.ndarray, cfg: dict) -> pd.Series:
    """
    แปลงความน่าจะเป็นที่ราคาจะขึ้น (proba_up) เป็นสัญญาณ  1 = ซื้อ, -1 = ขาย/ปิดสถานะ, 0 = ถือเดิม

      proba_up >= confidence_threshold        -> 1
      proba_up <= 1 - confidence_threshold    -> -1
      min_confidence_gap: บังคับให้ |proba_up - 0.5| >= gap ด้วย
        *หมายเหตุ*: เมื่อ threshold >= 0.5 + gap เงื่อนไขนี้ซ้ำซ้อนกับ threshold (เช่น 0.55 กับ gap 0.05
        เทียบเท่ากันพอดี) gap จะมีผลก็ต่อเมื่อ threshold ต่ำกว่า 0.5 + gap เท่านั้น
    """
    proba_up = np.asarray(proba_up, dtype=float)
    threshold = cfg["confidence_threshold"]
    gap = cfg.get("min_confidence_gap", 0.0)

    signal = pd.Series(0, index=range(len(proba_up)))
    signal[proba_up >= threshold] = 1
    signal[proba_up <= (1 - threshold)] = -1

    if gap > 0:
        confident = np.abs(proba_up - 0.5) >= gap
        signal[~confident] = 0

    return signal


def compute_dynamic_position_size(
    confidence: float, threshold: float, min_size: float, max_size: float
) -> float:
    """
    ขนาด position ตามความมั่นใจของโมเดล: ยิ่งมั่นใจสูงกว่า threshold ยิ่งลงทุนเยอะ
    (scale เชิงเส้นระหว่าง threshold ถึง 1.0) ใช้ตอนเข้า long เท่านั้น (ระบบ long-only)
    """
    span = 1.0 - threshold
    if span <= 0:
        return max_size
    frac = min(max((confidence - threshold) / span, 0.0), 1.0)
    return min_size + (max_size - min_size) * frac


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cfg: dict,
    confidence: pd.Series | None = None,
    close_at_end: bool = True,
) -> dict:
    """
    backtest แบบ long-only/flat:
      signal == 1  -> เข้า long (ถ้ายังไม่มีสถานะ)
      signal == -1 -> ปิดสถานะ (ถ้ามี long อยู่)
      signal == 0  -> ถือสถานะเดิม

    - stop loss / take profit เช็คจาก low/high ของวัน แล้ว exit ที่ราคา stop/TP เป๊ะ
      (ถ้าวันเดียวแตะทั้งคู่ ถือว่า stop loss โดนก่อน — สมมติฐานระมัดระวังสุด)
    - confidence: proba_up รายวัน ใช้คำนวณ dynamic sizing (rule-based ไม่มี -> fixed sizing เสมอ)
    - close_at_end: ปิดสถานะที่ค้างอยู่ด้วยราคาปิดแท่งสุดท้าย (พร้อมค่าธรรมเนียม) เพื่อให้สถิติเทรดครบ

    คืนค่า dict: equity_curve, trade_log, summary
    """
    close = df["btc_close"].reset_index(drop=True)
    low = df["btc_low"].reset_index(drop=True)
    high = df["btc_high"].reset_index(drop=True)
    dates = df.index
    signal = signal.reset_index(drop=True)

    capital = cfg["initial_capital"]
    fee = cfg["fee_pct"]
    slippage = cfg["slippage_pct"]
    stop_loss_pct = cfg["stop_loss_pct"]
    take_profit_pct = cfg["take_profit_pct"]
    position_size_pct = cfg["position_size_pct"]

    dynamic_mode = cfg.get("position_sizing_mode", "fixed") == "dynamic" and confidence is not None
    threshold = cfg.get("confidence_threshold", 0.5)
    min_size = cfg.get("min_position_size_pct", position_size_pct)
    max_size = cfg.get("max_position_size_pct", position_size_pct)
    if dynamic_mode:
        confidence = confidence.reset_index(drop=True)

    # cash และ shares เป็น state แยกกันชัดเจน (ห้ามคำนวณย้อนจาก equity รวม จะนับมูลค่าสถานะซ้ำ)
    equity = [capital]
    cash = capital
    position = 0
    entry_price = 0.0
    entry_idx = 0
    shares = 0.0
    days_in_market = 0
    trades: list[dict] = []

    def _close_position(i: int, raw_exit_price: float, change_pct: float, reason: str) -> None:
        nonlocal cash, position, shares
        exit_price = raw_exit_price * (1 - slippage)
        proceeds = shares * exit_price * (1 - fee)
        cost = shares * entry_price
        pnl = proceeds - cost
        cash += proceeds
        trades.append({
            "exit_date": dates[i], "exit_price": exit_price,
            "pnl": pnl, "pnl_pct": change_pct,            # pnl_pct = % ราคาเปลี่ยน (ก่อน fee/slippage ขาออก)
            "pnl_net_pct": pnl / cost if cost > 0 else 0.0,
            "holding_days": i - entry_idx,
            "reason": reason,
        })
        position = 0
        shares = 0.0

    for i in range(1, len(close)):
        price = close.iloc[i]
        sig = signal.iloc[i]

        if position == 1:
            low_change_pct = (low.iloc[i] - entry_price) / entry_price
            high_change_pct = (high.iloc[i] - entry_price) / entry_price
            close_change_pct = (price - entry_price) / entry_price

            stop_hit = low_change_pct <= -stop_loss_pct
            tp_hit = high_change_pct >= take_profit_pct

            if stop_hit:
                _close_position(i, entry_price * (1 - stop_loss_pct), -stop_loss_pct, "stop_loss")
            elif tp_hit:
                _close_position(i, entry_price * (1 + take_profit_pct), take_profit_pct, "take_profit")
            elif sig == -1:
                _close_position(i, price, close_change_pct, "signal")

        elif sig == 1:
            entry_price = price * (1 + slippage)
            entry_idx = i
            if dynamic_mode:
                size_pct = compute_dynamic_position_size(confidence.iloc[i], threshold, min_size, max_size)
            else:
                size_pct = position_size_pct
            invest_amount = cash * size_pct
            shares = (invest_amount * (1 - fee)) / entry_price
            cash -= invest_amount
            position = 1
            trades.append({
                "entry_date": dates[i], "entry_price": entry_price,
                "position_size_pct": size_pct,
            })

        if position == 1:
            days_in_market += 1
        # mark-to-market ของวันนี้ = cash + มูลค่าสถานะตามราคาตลาด
        equity.append(cash + (shares * price if position == 1 else 0.0))

    if close_at_end and position == 1:
        last_price = close.iloc[-1]
        _close_position(len(close) - 1, last_price, (last_price - entry_price) / entry_price, "end_of_data")
        equity[-1] = cash

    equity_series = pd.Series(equity, index=dates[: len(equity)])
    trade_log = pd.DataFrame(trades)

    exits = trade_log[trade_log["pnl"].notna()] if "pnl" in trade_log.columns else trade_log.iloc[0:0]
    n_trades = len(exits)
    total_return = (equity_series.iloc[-1] / capital) - 1

    if n_trades > 0:
        wins = exits[exits["pnl"] > 0]
        losses = exits[exits["pnl"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = wins["pnl"].mean() if len(wins) else 0.0
        avg_loss = losses["pnl"].mean() if len(losses) else 0.0
        gross_win = wins["pnl"].sum()
        gross_loss = losses["pnl"].sum()
        profit_factor = gross_win / abs(gross_loss) if gross_loss < 0 else np.nan
        avg_holding = exits["holding_days"].mean()
    else:
        win_rate = avg_win = avg_loss = 0.0
        profit_factor = np.nan
        avg_holding = 0.0

    running_max = equity_series.cummax()
    max_drawdown = ((equity_series - running_max) / running_max).min()

    daily_returns = equity_series.pct_change().dropna()
    ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    std = daily_returns.std()
    sharpe = (daily_returns.mean() / std) * ann if std > 0 else 0.0
    downside = np.sqrt((np.minimum(daily_returns, 0.0) ** 2).mean()) if len(daily_returns) else 0.0
    sortino = (daily_returns.mean() / downside) * ann if downside > 0 else 0.0

    n_days = max(len(equity_series) - 1, 1)
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1 if total_return > -1 else -1.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    summary = {
        "n_trades": n_trades,
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "exposure_pct": days_in_market / n_days * 100,
        "avg_holding_days": avg_holding,
        "final_equity": equity_series.iloc[-1],
    }

    if n_trades == 0:
        logger.warning(
            "*** ไม่มีเทรดเกิดขึ้นเลย — เช็ค confidence_threshold ใน plan.yaml ว่าต่ำพอหรือยัง "
            "หรือเช็คว่า predict_proba กระจายค่าสมเหตุสมผลไหม (ไม่ใช่กระจุกที่ 0.5) ***"
        )

    return {"equity_curve": equity_series, "trade_log": trade_log, "summary": summary}


def run_buy_and_hold(df: pd.DataFrame, cfg: dict) -> dict:
    """Buy & Hold บนช่วงเดียวกัน (ลงทุนเต็ม 100% ครั้งเดียว ไม่มี SL/TP) ผ่าน engine เดียวกันเพื่อให้เทียบแฟร์"""
    bh_cfg = {
        **cfg,
        "stop_loss_pct": 1e9, "take_profit_pct": 1e9,
        "position_size_pct": 1.0, "position_sizing_mode": "fixed",
    }
    return run_backtest(df, pd.Series(1, index=range(len(df))), bh_cfg)


def compare_strategies(
    df: pd.DataFrame,
    signals: dict[str, pd.Series],
    cfg: dict,
    confidence_map: dict[str, pd.Series] | None = None,
    include_buy_and_hold: bool = True,
) -> pd.DataFrame:
    """
    รัน backtest ให้ทุกกลยุทธ์ใน dict แล้วสรุปผลเทียบกันเป็นตาราง (+ แถว Buy_and_Hold เป็นเส้นฐาน)
    confidence_map: {ชื่อ strategy: proba_up} ใช้ dynamic sizing เฉพาะ strategy ที่มีใน dict นี้
    """
    confidence_map = confidence_map or {}
    rows = []
    for name, sig in signals.items():
        result = run_backtest(df, sig.reset_index(drop=True), cfg, confidence=confidence_map.get(name))
        rows.append({"strategy": name, **result["summary"]})
    if include_buy_and_hold:
        rows.append({"strategy": "Buy_and_Hold", **run_buy_and_hold(df, cfg)["summary"]})
    return pd.DataFrame(rows).set_index("strategy")
