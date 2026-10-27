"""
generate_report.py — สร้างตาราง กราฟ และสรุปผลสำหรับใส่ในเล่มสหกิจ

รัน:  python generate_report.py     หรือ     python -m main report
ต้องรัน fetch-data และ train มาก่อน  ผลลัพธ์ทั้งหมดอยู่ที่ results/report/

สิ่งที่สร้าง:
  strategy_comparison.csv/.md   ML vs rule-based vs Buy&Hold (ช่วง out-of-sample เดียวกัน)
  classification_metrics.json   accuracy / AUC / Brier / p-value ของการทำนายทิศทาง
  calibration.csv/.png          proba ที่โมเดลให้ vs อัตราที่ราคาขึ้นจริง
  segment_stability.csv         ผลแยกเป็น 4 ช่วงเวลา (ดูว่าชนะเพราะช่วงใดช่วงหนึ่งหรือไม่)
  sensitivity_sharpe.csv/.png   Sharpe เมื่อขยับ threshold x stop-loss (ความทนทานต่อพารามิเตอร์)
  bootstrap: ช่วงความเชื่อมั่น 95% ของ Sharpe (block bootstrap)
  exit_reasons.csv/.png, equity_curves.png, drawdown.png, monthly_returns.png,
  price_with_trades.png, walk_forward_accuracy.png, feature_importance.png/csv
  REPORT_SUMMARY.md             สรุปภาษาไทยพร้อมตัวเลขจริงและข้อควรระวัง
  key_metrics.json              ตัวเลขสำคัญทั้งหมดในไฟล์เดียว
(ป้ายกำกับในกราฟเป็นภาษาอังกฤษ เพราะฟอนต์เริ่มต้นของ matplotlib บน Windows ไม่รองรับภาษาไทย)
"""
from __future__ import annotations

import json
import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.backtester import TRADING_DAYS_PER_YEAR, run_backtest, run_buy_and_hold, signals_from_confidence
from src.evaluation import load_oos, segment_slices
from src.paths import FEATURE_DATA_PATH, OOS_PROBA_PATH, REPORT_DIR, RESULTS_DIR, WF_RESULTS_PATH
from src.rule_strategies import get_all_strategies
from src.utils import get_logger, load_config

logger = get_logger("generate_report")

COLORS = {
    "ML_Ensemble": "#1b9e77", "Buy_and_Hold": "#7f7f7f", "RSI_Reversal": "#d95f02",
    "MACD_Crossover": "#7570b3", "Volatility_Breakout": "#e7298a",
}
SUMMARY_COLS = [
    "n_trades", "total_return_pct", "annual_return_pct", "win_rate_pct", "profit_factor",
    "max_drawdown_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "exposure_pct", "avg_holding_days",
]


# ---------------------------------------------------------------- helpers
def _num(v, nd=2, suffix=""):
    return "-" if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f"{v:,.{nd}f}{suffix}"


INT_COLS = {"n_trades", "n", "segment"}  # คอลัมน์จำนวนนับ แสดงเป็นจำนวนเต็ม


def df_to_markdown(df: pd.DataFrame, index_name: str = "") -> str:
    cols = [index_name or (df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for col, v in row.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                cells.append(_num(v, 0) if col in INT_COLS else _num(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _style(ax):
    ax.grid(True, alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(REPORT_DIR / name, dpi=150)
    plt.close(fig)


def binom_p_greater(k: int, n: int, p0: float) -> float:
    """P(X >= k) เมื่อ X ~ Binomial(n, p0)"""
    if n <= 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p0, alternative="greater").pvalue)
    except Exception:
        z = (k - 0.5 - n * p0) / math.sqrt(n * p0 * (1 - p0))
        return 0.5 * math.erfc(z / math.sqrt(2))


def block_bootstrap_sharpe(daily_returns: np.ndarray, n_boot: int = 2000, block: int = 7, seed: int = 42):
    """ช่วงความเชื่อมั่น 95% ของ Sharpe ด้วย moving-block bootstrap (คง autocorrelation ระยะสั้นของผลตอบแทน)"""
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    if n < block * 3:
        return (float("nan"), float("nan")), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        sharpes[b] = sample.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else 0.0
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return (float(lo), float(hi)), float((sharpes > 0).mean())


# ---------------------------------------------------------------- sections
def classification_metrics(df_oos: pd.DataFrame, proba: np.ndarray, threshold: float, horizon: int) -> dict:
    y = df_oos["target"].astype(int).to_numpy()
    pred = (proba >= 0.5).astype(int)
    n, k = len(y), int((pred == y).sum())
    base_rate = float(max(y.mean(), 1 - y.mean()))
    conf = (proba >= threshold) | (proba <= 1 - threshold)
    buy = proba >= threshold

    # label ทับซ้อนกัน horizon วัน -> ตัวอย่างรายวันไม่อิสระ ใช้ n_eff = n / horizon เป็นค่าแบบระมัดระวัง
    n_eff = max(n // max(horizon, 1), 1)
    k_eff = int(round(k / n * n_eff))
    return {
        "n_oos": n,
        "accuracy": k / n,
        "majority_class_baseline": base_rate,
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(y, proba)),
        "p_value_vs_50pct_naive": binom_p_greater(k, n, 0.5),
        "p_value_vs_majority_naive": binom_p_greater(k, n, base_rate),
        "p_value_vs_majority_conservative": binom_p_greater(k_eff, n_eff, base_rate),
        "n_effective_independent": n_eff,
        "n_confident_signals": int(conf.sum()),
        "accuracy_when_confident": float((pred[conf] == y[conf]).mean()) if conf.any() else float("nan"),
        "n_buy_signals": int(buy.sum()),
        "up_rate_after_buy_signal": float(y[buy].mean()) if buy.any() else float("nan"),
    }


def calibration_table(df_oos: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    y = df_oos["target"].astype(int).to_numpy()
    bins = pd.cut(proba, [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0], include_lowest=True)
    tab = (pd.DataFrame({"bin": bins, "proba": proba, "up": y})
           .groupby("bin", observed=True)
           .agg(n=("up", "size"), mean_predicted=("proba", "mean"), actual_up_rate=("up", "mean")))
    tab.index = tab.index.astype(str)
    tab.index.name = "proba_bin"
    return tab


def segment_table(df_oos, ml_signal, proba, cfg, k=4) -> pd.DataFrame:
    rows = []
    for j, seg in enumerate(segment_slices(len(df_oos), k), start=1):
        d = df_oos.iloc[seg]
        if len(d) < 30:
            continue
        ml = run_backtest(d, ml_signal.iloc[seg].reset_index(drop=True), cfg, confidence=pd.Series(proba[seg]))["summary"]
        bh = run_buy_and_hold(d, cfg)["summary"]
        rows.append({
            "segment": j, "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
            "ml_return_pct": ml["total_return_pct"], "ml_sharpe": ml["sharpe_ratio"], "ml_max_dd_pct": ml["max_drawdown_pct"],
            "bh_return_pct": bh["total_return_pct"], "bh_sharpe": bh["sharpe_ratio"], "bh_max_dd_pct": bh["max_drawdown_pct"],
            "n_trades": ml["n_trades"],
        })
    return pd.DataFrame(rows).set_index("segment")


def sensitivity_grid(df_oos, proba, cfg) -> pd.DataFrame:
    thresholds = [0.52, 0.55, 0.58, 0.60, 0.65]
    stop_losses = [0.02, 0.03, 0.05]
    grid = pd.DataFrame(index=[f"SL={s:.0%}" for s in stop_losses], columns=[f"thr={t:.2f}" for t in thresholds], dtype=float)
    for s in stop_losses:
        for t in thresholds:
            c = {**cfg, "confidence_threshold": t, "stop_loss_pct": s}
            sig = signals_from_confidence(proba, c)
            grid.loc[f"SL={s:.0%}", f"thr={t:.2f}"] = run_backtest(df_oos, sig, c, confidence=pd.Series(proba))["summary"]["sharpe_ratio"]
    return grid


# ---------------------------------------------------------------- plots
def plot_equity(results: dict, period: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        eq = res["equity_curve"]
        ax.plot(eq.index, eq.values, label=name, color=COLORS.get(name), lw=2.4 if name == "ML_Ensemble" else 1.2,
                ls="--" if name == "Buy_and_Hold" else "-")
    ax.set_title(f"Equity curves, out-of-sample ({period})")
    ax.set_ylabel("Equity (USD)")
    ax.legend(frameon=False, ncol=2)
    _style(ax)
    _save(fig, "equity_curves.png")


def plot_drawdown(results: dict):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, color in (("ML_Ensemble", COLORS["ML_Ensemble"]), ("Buy_and_Hold", COLORS["Buy_and_Hold"])):
        eq = results[name]["equity_curve"]
        dd = (eq / eq.cummax() - 1) * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.35, color=color, label=name)
    ax.set_title("Drawdown (%)")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "drawdown.png")


def plot_monthly(results: dict):
    def monthly(eq: pd.Series) -> pd.Series:
        try:
            m = eq.resample("ME").last()
        except ValueError:  # pandas < 2.2
            m = eq.resample("M").last()
        r = m.pct_change() * 100
        r.iloc[0] = (m.iloc[0] / eq.iloc[0] - 1) * 100
        return r
    ml, bh = monthly(results["ML_Ensemble"]["equity_curve"]), monthly(results["Buy_and_Hold"]["equity_curve"])
    x = np.arange(len(ml))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, ml.values, 0.4, label="ML_Ensemble", color=COLORS["ML_Ensemble"])
    ax.bar(x + 0.2, bh.reindex(ml.index).values, 0.4, label="Buy_and_Hold", color=COLORS["Buy_and_Hold"])
    ax.set_xticks(x[:: max(len(x) // 12, 1)])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in ml.index[:: max(len(x) // 12, 1)]], rotation=45)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Monthly return (%)")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "monthly_returns.png")


def plot_calibration(tab: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.3, 0.7], [0.3, 0.7], ls="--", color="gray", label="perfect calibration")
    ax.scatter(tab["mean_predicted"], tab["actual_up_rate"], s=40 + tab["n"] / tab["n"].max() * 260,
               color=COLORS["ML_Ensemble"], alpha=0.8, label="bins (size = n)")
    ax.set_xlabel("Mean predicted P(up)")
    ax.set_ylabel("Actual up rate")
    ax.set_title("Calibration (out-of-sample)")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "calibration.png")


def plot_exit_reasons(exits: pd.DataFrame):
    summ = exits.groupby("reason").agg(n=("pnl", "size"), total_pnl=("pnl", "sum"), avg_pnl=("pnl", "mean")).round(2)
    summ.to_csv(REPORT_DIR / "exit_reasons.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    colors = ["#2ca25f" if v >= 0 else "#de2d26" for v in summ["total_pnl"]]
    axes[0].bar(summ.index, summ["n"], color="#5b8db8")
    axes[0].set_title("Trades by exit reason")
    axes[1].bar(summ.index, summ["total_pnl"], color=colors)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("Total PnL by exit reason (USD)")
    for ax in axes:
        _style(ax)
    _save(fig, "exit_reasons.png")
    return summ


def plot_price_with_trades(df_oos: pd.DataFrame, trade_log: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df_oos.index, df_oos["btc_close"], color="#f2a900", lw=1.2, label="BTC close")
    if "entry_date" in trade_log:
        e = trade_log[trade_log["entry_date"].notna()]
        ax.scatter(e["entry_date"], e["entry_price"], marker="^", color="#2ca25f", s=36, label="entry", zorder=3)
    if "exit_date" in trade_log:
        x = trade_log[trade_log["exit_date"].notna()]
        ax.scatter(x["exit_date"], x["exit_price"], marker="v", color="#de2d26", s=36, label="exit", zorder=3)
    ax.set_title("BTC price with ML entries/exits (out-of-sample)")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "price_with_trades.png")


def plot_sensitivity(grid: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    im = ax.imshow(grid.values.astype(float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Sharpe vs threshold x stop-loss")
    fig.colorbar(im, ax=ax)
    _save(fig, "sensitivity_sharpe.png")


def plot_walk_forward(wf: pd.DataFrame):
    need = {"fold", "train_accuracy", "val_accuracy", "accuracy"}
    if not need.issubset(wf.columns):
        return
    x = np.arange(len(wf))
    fig, ax = plt.subplots(figsize=(8, 4))
    for off, col, label, color in ((-0.27, "train_accuracy", "Train", "#3498db"), (0, "val_accuracy", "Val", "#9b59b6"),
                                   (0.27, "accuracy", "OOS (test)", "#e67e22")):
        ax.bar(x + off, wf[col] * 100, 0.27, label=label, color=color)
    ax.axhline(50, ls="--", color="gray")
    ax.set_xticks(x, [f"Fold {int(f)}" for f in wf["fold"]])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Walk-forward: train vs validation vs out-of-sample")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "walk_forward_accuracy.png")


# ---------------------------------------------------------------- main
def main():
    config = load_config()
    cfg = config["backtest"]
    horizon = config["model"]["target_horizon"]

    if not FEATURE_DATA_PATH.exists() or not OOS_PROBA_PATH.exists():
        raise SystemExit("ไม่พบ feature_matrix.csv หรือ oos_proba.csv — รัน `python -m main fetch-data` และ `python -m main train` ก่อน")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
    df_oos, proba = load_oos(df, OOS_PROBA_PATH)
    period = f"{df_oos.index[0].date()} to {df_oos.index[-1].date()}"
    ml_signal = signals_from_confidence(proba, cfg)

    # --- backtest ทุกกลยุทธ์ + Buy & Hold บนช่วงเดียวกัน
    signals = get_all_strategies(df_oos)
    signals["ML_Ensemble"] = ml_signal
    results = {}
    for name, sig in signals.items():
        results[name] = run_backtest(df_oos, sig.reset_index(drop=True), cfg,
                                     confidence=pd.Series(proba) if name == "ML_Ensemble" else None)
    results["Buy_and_Hold"] = run_buy_and_hold(df_oos, cfg)
    comparison = pd.DataFrame({n: r["summary"] for n, r in results.items()}).T
    comparison.index.name = "strategy"
    comp_view = comparison[SUMMARY_COLS].astype(float)
    comp_view.round(3).to_csv(REPORT_DIR / "strategy_comparison.csv")
    (REPORT_DIR / "strategy_comparison.md").write_text(df_to_markdown(comp_view, "strategy"), encoding="utf-8")

    # --- ความสามารถในการทำนาย
    clf = classification_metrics(df_oos, proba, cfg["confidence_threshold"], horizon)
    (REPORT_DIR / "classification_metrics.json").write_text(json.dumps(clf, indent=2), encoding="utf-8")
    calib = calibration_table(df_oos, proba)
    calib.round(4).to_csv(REPORT_DIR / "calibration.csv")

    # --- ความทนทาน
    ml_ret = results["ML_Ensemble"]["equity_curve"].pct_change().dropna().to_numpy()
    bh_ret = results["Buy_and_Hold"]["equity_curve"].pct_change().dropna().to_numpy()
    ml_ci, ml_p_pos = block_bootstrap_sharpe(ml_ret, block=max(horizon, 7))
    bh_ci, _ = block_bootstrap_sharpe(bh_ret, block=max(horizon, 7))
    seg = segment_table(df_oos, ml_signal, proba, cfg)
    seg.round(2).to_csv(REPORT_DIR / "segment_stability.csv")
    grid = sensitivity_grid(df_oos, proba, cfg)
    grid.round(3).to_csv(REPORT_DIR / "sensitivity_sharpe.csv")

    # --- กราฟ
    plot_equity(results, period)
    plot_drawdown(results)
    plot_monthly(results)
    plot_calibration(calib)
    plot_sensitivity(grid)
    trade_log = results["ML_Ensemble"]["trade_log"]
    trade_log.to_csv(REPORT_DIR / "ml_trade_log.csv", index=False)
    exit_summary = None
    if "pnl" in trade_log.columns and trade_log["pnl"].notna().any():
        exits = trade_log[trade_log["pnl"].notna()]
        exit_summary = plot_exit_reasons(exits)
    plot_price_with_trades(df_oos, trade_log)
    if WF_RESULTS_PATH.exists():
        wf = pd.read_csv(WF_RESULTS_PATH)
        plot_walk_forward(wf)
        wf.round(4).to_csv(REPORT_DIR / "walk_forward_results.csv", index=False)
    else:
        wf = None

    # feature importance (ไม่บังคับ — ข้ามได้ถ้าโมเดลไม่ตรงกับ feature ปัจจุบัน)
    try:
        import feature_importance
        feature_importance.main()
        for fname in ("feature_importance.png", "feature_importance.csv"):
            if (RESULTS_DIR / fname).exists():
                shutil.copy(RESULTS_DIR / fname, REPORT_DIR / fname)
    except (SystemExit, Exception) as e:  # noqa: BLE001
        logger.warning(f"ข้าม feature importance: {e}")

    # --- สรุปภาษาไทย
    ml = comparison.loc["ML_Ensemble"]
    bh = comparison.loc["Buy_and_Hold"]
    rules = comparison.drop(index=["ML_Ensemble", "Buy_and_Hold"])
    best_rule = rules["sharpe_ratio"].astype(float).idxmax() if len(rules) else None
    notes = []
    ml_sh, bh_sh = float(ml["sharpe_ratio"]), float(bh["sharpe_ratio"])
    if best_rule:
        rule_sh = float(rules.loc[best_rule, "sharpe_ratio"])
        if rule_sh > ml_sh + 0.05:
            notes.append(f"**ML Ensemble ไม่ใช่กลยุทธ์ที่ Sharpe ดีที่สุด** — {best_rule} ให้ Sharpe {rule_sh:.2f} สูงกว่า ML ({ml_sh:.2f}) ควรรายงานตามจริง")
        elif abs(rule_sh - ml_sh) <= 0.05:
            notes.append(f"Sharpe ของ ML ({ml_sh:.2f}) ใกล้เคียง {best_rule} ({rule_sh:.2f}) มากจนแยกไม่ออกทางสถิติ — "
                         "ความซับซ้อนของโมเดลไม่ได้สร้างความได้เปรียบเหนือ baseline ง่ายๆ")
    if bh_sh > ml_sh:
        notes.append(f"**Buy & Hold ให้ผลปรับความเสี่ยงดีกว่า ML**: Sharpe {bh_sh:.2f} เทียบ {ml_sh:.2f}, "
                     f"Calmar {bh['calmar_ratio']:.2f} เทียบ {ml['calmar_ratio']:.2f} (Sharpe ไม่ขึ้นกับสัดส่วนเงินลงทุน จึงไม่ใช่แค่เพราะ ML ถือสถานะ {ml['exposure_pct']:.0f}% ของเวลา) "
                     f"ข้อดีที่เหลือของ ML คือ Max Drawdown ต่ำกว่า ({ml['max_drawdown_pct']:.1f}% เทียบ {bh['max_drawdown_pct']:.1f}%) ส่วนผลตอบแทนรวมคือ "
                     f"{ml['total_return_pct']:.1f}% เทียบ {bh['total_return_pct']:.1f}% (ถ้า BTC ขาขึ้นแรงในช่วงข้อมูล กลยุทธ์ long-only ที่ถือสถานะบางช่วงจะเสียเปรียบเชิงโครงสร้าง)")
    elif float(ml["total_return_pct"]) < float(bh["total_return_pct"]):
        notes.append(f"ผลตอบแทนรวมของ ML ({ml['total_return_pct']:.1f}%) ต่ำกว่า Buy & Hold ({bh['total_return_pct']:.1f}%) "
                     f"แต่ปรับความเสี่ยงแล้วดีกว่า (Sharpe {ml_sh:.2f} เทียบ {bh_sh:.2f}, Max DD {ml['max_drawdown_pct']:.1f}% เทียบ {bh['max_drawdown_pct']:.1f}%)")
    if ml_ci[0] <= 0 <= ml_ci[1]:
        notes.append(f"ช่วงความเชื่อมั่น 95% ของ Sharpe ({ml_ci[0]:.2f} ถึง {ml_ci[1]:.2f}) ครอบคลุม 0 — "
                     "ยังสรุปไม่ได้ว่ากลยุทธ์มี edge จริงทางสถิติ")
    if len(seg) and (seg["ml_return_pct"] <= 0).any():
        bad = ", ".join(str(i) for i in seg.index[seg["ml_return_pct"] <= 0])
        notes.append(f"ผลไม่สม่ำเสมอข้ามช่วงเวลา: ช่วงที่ {bad} (จาก {len(seg)} ช่วง) ML ขาดทุนหรือเสมอตัว")
    if clf["p_value_vs_majority_conservative"] > 0.05:
        notes.append(f"accuracy {clf['accuracy']:.1%} ไม่ชนะการเดาด้วยคลาสส่วนใหญ่ ({clf['majority_class_baseline']:.1%}) "
                     f"อย่างมีนัยสำคัญ (p={clf['p_value_vs_majority_conservative']:.2f} แบบระมัดระวัง) — "
                     "ถ้ากลยุทธ์ทำกำไรได้ ผลนั้นมาจากการจัดการ position/ความเสี่ยงมากกว่าความแม่นยำของการทำนายทิศทาง")

    md = f"""# สรุปผลการทดลอง (สร้างอัตโนมัติจาก generate_report.py)

ช่วง out-of-sample: **{period}** ({len(df_oos)} วัน) | target_horizon = {horizon} วัน | purge = {config['model']['walk_forward'].get('purge_days', horizon)} วัน
พารามิเตอร์ backtest: threshold={cfg['confidence_threshold']}, stop_loss={cfg['stop_loss_pct']:.0%}, take_profit={cfg['take_profit_pct']:.0%}, \
sizing={cfg['position_sizing_mode']}, fee={cfg['fee_pct']:.2%}, slippage={cfg['slippage_pct']:.2%}

## 1) เปรียบเทียบกลยุทธ์ (ช่วงเดียวกัน, engine เดียวกัน)
{df_to_markdown(comp_view, 'strategy')}

## 2) ความสามารถในการทำนายทิศทาง ({horizon} วันข้างหน้า)
- Accuracy {clf['accuracy']:.1%} (baseline คลาสส่วนใหญ่ {clf['majority_class_baseline']:.1%}) | ROC-AUC {_num(clf['roc_auc'], 3)} | Brier {_num(clf['brier_score'], 3)}
- p-value เทียบ baseline: {clf['p_value_vs_majority_naive']:.3f} (แบบไม่ปรับ) / {clf['p_value_vs_majority_conservative']:.3f} (ปรับ label ทับซ้อน, n_eff={clf['n_effective_independent']})
- เฉพาะวันที่โมเดลมั่นใจ (เกิน threshold): {clf['n_confident_signals']} วัน accuracy {_num(clf['accuracy_when_confident'] * 100 if clf['n_confident_signals'] else float('nan'), 1, '%')}
- หลังสัญญาณซื้อ ({clf['n_buy_signals']} วัน) ราคาขึ้นจริง {_num(clf['up_rate_after_buy_signal'] * 100 if clf['n_buy_signals'] else float('nan'), 1, '%')}

## 3) ความทนทานของผล
- Sharpe ของ ML: {ml['sharpe_ratio']:.2f} (95% CI {ml_ci[0]:.2f} ถึง {ml_ci[1]:.2f}, P(Sharpe>0) = {ml_p_pos:.0%}) | Buy & Hold: {bh['sharpe_ratio']:.2f} (95% CI {bh_ci[0]:.2f} ถึง {bh_ci[1]:.2f})
- แยกเป็น 4 ช่วงเวลา:

{df_to_markdown(seg.round(2), 'segment')}

- ความไวต่อพารามิเตอร์ (Sharpe):

{df_to_markdown(grid.round(2), 'stop-loss')}

## 4) ข้อสังเกตที่ควรเขียนในเล่มตามจริง
{chr(10).join('- ' + n for n in notes) if notes else '- ไม่พบข้อสังเกตเชิงลบจากการตรวจอัตโนมัติ (ยังควรอ่านข้อจำกัดด้านล่างประกอบ)'}

## 5) ข้อจำกัด (ต้องระบุในเล่มเสมอ)
- threshold / stop-loss / take-profit ถูกเลือกจาก sweep บนผล out-of-sample ชุดเดียวกับที่รายงาน จึงมี selection bias — ตัวเลขคือ "ผลเมื่อเลือกพารามิเตอร์ดีที่สุดย้อนหลัง" ไม่ใช่ผลที่คาดหวังในอนาคต การยืนยันจริงต้องดู forward test
- ข้อมูลรายวันจาก Binance/Yahoo ไม่ใช่ tick-level; backtest ไม่จำลอง order book, partial fill, latency
- ตัวอย่างรายวันของ label {horizon} วันทับซ้อนกัน → จำนวนตัวอย่างอิสระจริงน้อยกว่าจำนวนแถวมาก
- ผลย้อนหลังไม่การันตีอนาคต โดยเฉพาะ crypto ที่ regime เปลี่ยนเร็ว ต้อง monitor และ retrain เป็นระยะ
"""
    (REPORT_DIR / "REPORT_SUMMARY.md").write_text(md, encoding="utf-8")

    key = {
        "period": period, "n_oos_days": len(df_oos),
        "strategies": {n: {k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else float(v))
                           for k, v in comp_view.loc[n].items()} for n in comp_view.index},
        "classification": clf, "ml_sharpe_ci95": list(ml_ci), "ml_p_sharpe_positive": ml_p_pos,
        "bh_sharpe_ci95": list(bh_ci), "notes": notes,
    }
    (REPORT_DIR / "key_metrics.json").write_text(json.dumps(key, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"สร้างรายงานเสร็จ -> {REPORT_DIR}")
    print(comp_view.round(2).to_string())
    for n in notes:
        print(" *", n)


if __name__ == "__main__":
    main()
