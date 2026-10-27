"""
app.py — Streamlit dashboard สำหรับดูผลลัพธ์ pipeline
รันด้วย: python -m streamlit run app.py
(ห้ามรันด้วย python app.py ตรงๆ เดี๋ยวจะเจอ error "missing ScriptRunContext")
"""
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import discord_alert
from src import features
from src.backtester import compare_strategies, run_backtest, signals_from_confidence
from src.evaluation import load_oos
from src.live_signal import get_latest_signal, load_live_frame
from src.model import EnsembleModel
from src.rule_strategies import get_all_strategies
from src.utils import PROJECT_ROOT, load_config

st.set_page_config(page_title="BTC/USDT Trading Pipeline", layout="wide")

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feature_matrix.csv"
OOS_PROBA_PATH = PROJECT_ROOT / "data" / "oos_proba.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_model.joblib"
WF_RESULTS_PATH = PROJECT_ROOT / "results" / "walk_forward_results.csv"

SWEEP_THRESHOLDS = [round(0.50 + i * 0.01, 2) for i in range(46)]  # 0.50 - 0.95 เหมือน sweep_threshold.py
MIN_TRADES_FOR_BEST = 20  # ต้องมีเทรดอย่างน้อยเท่านี้ ถึงจะนับว่า threshold นั้นน่าเชื่อถือ


@st.cache_data(show_spinner=False)
def run_threshold_sweep(_df_oos, _proba_up, backtest_cfg_items: tuple, data_key: str) -> pd.DataFrame:
    """
    รัน backtest ซ้ำทุก threshold (0.50-0.95) บน OOS proba ชุดเดียวกัน — logic เดียวกับ sweep_threshold.py
    _df_oos/_proba_up ขึ้นต้นด้วย _ เพื่อไม่ให้ Streamlit เสียเวลา hash; ใช้ data_key + cfg เป็น cache key แทน
    """
    base_cfg = dict(backtest_cfg_items)
    rows = []
    # threshold สูงๆ จะไม่มีเทรดเลยเป็นเรื่องปกติของ sweep — ปิด warning "ไม่มีเทรด" ชั่วคราว ไม่ให้ท่วม terminal/log
    bt_logger = logging.getLogger("backtester")
    prev_level = bt_logger.level
    bt_logger.setLevel(logging.ERROR)
    try:
        for t in SWEEP_THRESHOLDS:
            cfg_t = {**base_cfg, "confidence_threshold": t}
            sig = signals_from_confidence(_proba_up, cfg_t)
            res = run_backtest(_df_oos, sig, cfg_t, confidence=pd.Series(_proba_up))["summary"]
            rows.append({
                "threshold": t,
                "n_trades": res["n_trades"],
                "total_return_pct": res["total_return_pct"],
                "sharpe_ratio": res["sharpe_ratio"],
                "max_drawdown_pct": res["max_drawdown_pct"],
            })
    finally:
        bt_logger.setLevel(prev_level)
    return pd.DataFrame(rows)

st.title("📊 BTC/USDT Trading Pipeline Dashboard")

config = load_config()

if not FEATURE_DATA_PATH.exists():
    st.error(
        "ยังไม่มีข้อมูล — รันคำสั่งนี้ใน terminal ก่อน:\n\n"
        "`python -m main fetch-data`\n\n"
        "`python -m main train`"
    )
    st.stop()

df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
live_df = load_live_frame()  # ถึงแท่งล่าสุดจริง ใช้กับสัญญาณล่าสุดและการ์ดราคาด้านล่าง

# --- Live Signal (สัญญาณล่าสุด ณ ตอนนี้) ---
# ใช้ logic เดียวกับ discord_alert.py ผ่านโมดูลกลาง src/live_signal.py และข้อมูลจาก live_features.csv
# (ยาวถึงแท่งที่ปิดล่าสุดจริง ต่างจาก feature_matrix.csv ที่ถูกตัดท้าย target_horizon แถว)
live_signal = None
if MODEL_PATH.exists():
    live_model = EnsembleModel.load(MODEL_PATH)
    try:
        live_signal = get_latest_signal(live_df, live_model, config)
    except Exception as exc:  # โมเดล/ฟีเจอร์ไม่ตรงกัน ฯลฯ -> แจ้งให้ชัด แต่ไม่ให้ทั้ง dashboard พัง
        st.error(f"คำนวณสัญญาณล่าสุดไม่ได้: {exc}")
else:
    st.warning("⚠️ ยังไม่มีโมเดลที่เทรนไว้ — รัน `python -m main train` ก่อนเพื่อดูสัญญาณล่าสุด")

if live_signal is not None:
    signal_style = {
        "BUY": ("🟢", "#2ECC71", "เข้าซื้อ"),
        "SELL": ("🔴", "#E74C3C", "ระวังขาลง"),
        "HOLD": ("🟡", "#F1C40F", "ยังไม่ชัดเจน"),
    }
    emoji, color, label_th = signal_style[live_signal["signal_label"]]

    st.markdown(
        f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 16px; margin-bottom: 20px;">
            <span style="font-size: 1.3rem; font-weight: 700; color: {color};">
                {emoji} สัญญาณล่าสุด: {live_signal['signal_label']} / {label_th}
            </span>
            <br>
            <span style="opacity: 0.8;">
                ข้อมูล ณ วันที่ {live_signal['as_of_date'].strftime('%Y-%m-%d')} •
                ความน่าจะเป็นขาขึ้น ({live_signal['horizon']} วันข้างหน้า): {live_signal['confidence_pct']:.1f}% •
                ความผันผวน 7 วัน: {live_signal['risk_emoji']} {live_signal['risk_level']} ({live_signal['volatility_7d_pct']:+.2f}%)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if live_signal["is_stale"]:
        st.warning(
            f"⚠️ ข้อมูลล่าสุดค้างอยู่ {live_signal['data_age_days']} วัน — "
            "รัน `python -m main fetch-data` เพื่ออัปเดตก่อนใช้สัญญาณนี้"
        )

# --- ปุ่มทดสอบส่ง Discord Alert ---
if live_signal is not None:
    if st.button("🔔 ทดสอบส่ง Discord Alert"):
        webhook_url = discord_alert.get_webhook_url(config)
        if not webhook_url:
            st.error(
                "ไม่พบ Discord Webhook URL — ตั้งค่าก่อนด้วย "
                "`$env:DISCORD_WEBHOOK_URL = \"...\"` แล้วรีสตาร์ท dashboard ใหม่ "
                "(streamlit ต้องเปิดใหม่ถึงจะเห็น environment variable ที่เพิ่งตั้ง)"
            )
        else:
            embed = discord_alert.build_embed(live_signal)
            with st.spinner("กำลังส่งแจ้งเตือนไป Discord..."):
                success = discord_alert.send_discord_alert(webhook_url, embed)
            if success:
                st.success(
                    f"ส่งแจ้งเตือนสำเร็จ — Prob Up: {live_signal['confidence_pct']:.1f}%, "
                    f"ราคาล่าสุด: ${live_signal['current_price']:,.2f}"
                )
            else:
                st.error("ส่งแจ้งเตือนไม่สำเร็จ — เช็ค Webhook URL ว่ายังใช้งานได้อยู่ไหม")

# --- Sidebar controls ---
st.sidebar.header("⚙️ ตั้งค่า Backtest")
threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50, max_value=0.95,
    value=float(config["backtest"]["confidence_threshold"]),
    step=0.01,
    help="ค่านี้ต่ำไปจะเทรดถี่เกิน, สูงไปจะไม่มีเทรดเลย — ลองปรับดูผลกระทบแบบ real-time",
)
config["backtest"]["confidence_threshold"] = threshold

# --- Price overview ---
st.subheader("ราคา BTC/USDT ล่าสุด")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ราคาปิดล่าสุด", f"${live_df['btc_close'].iloc[-1]:,.0f}")
col2.metric("เปลี่ยนแปลงรายวัน", f"{live_df['return_1d'].iloc[-1] * 100:+.2f}%")
col3.metric("RSI", f"{live_df['rsi'].iloc[-1]:.1f}")
regime_map = {1: "🟢 Bull", 0: "🟡 Sideways", -1: "🔴 Bear"}
col4.metric("Regime ปัจจุบัน", regime_map.get(live_df["regime"].iloc[-1], "N/A"))

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=live_df.index, y=live_df["btc_close"], name="BTC Close", line=dict(color="#f2a900")))
fig_price.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="ราคา (USD)")
st.plotly_chart(fig_price, use_container_width=True)

# --- Regime distribution ---
with st.expander("📈 Regime Classification"):
    regime_counts = df["regime"].map(regime_map).value_counts()
    fig_regime = go.Figure()
    fig_regime.add_trace(go.Bar(
        x=regime_counts.index.tolist(),
        y=regime_counts.values.tolist(),
        marker_color=["#2ecc71", "#f1c40f", "#e74c3c"][:len(regime_counts)],
    ))
    fig_regime.update_layout(height=300, margin=dict(t=20, b=20), yaxis_title="จำนวนวัน")
    st.plotly_chart(fig_regime, use_container_width=True)

# --- OOS Backtest results ---
st.subheader("🤖 ผลลัพธ์ ML Ensemble (Out-of-Sample)")

if OOS_PROBA_PATH.exists():
    df_oos, proba_up = load_oos(df, OOS_PROBA_PATH)  # จับคู่ด้วยวันที่ ไม่พังแม้ feature matrix ยาวขึ้นหลัง fetch-data

    signal = signals_from_confidence(proba_up, config["backtest"])
    result = run_backtest(df_oos, signal, config["backtest"], confidence=pd.Series(proba_up))
    summary = result["summary"]

    # Buy & Hold บนช่วง OOS เดียวกัน (ลงทุนเต็มจำนวนตั้งแต่วันแรก) ไว้เป็นเส้นเทียบ
    initial_capital = config["backtest"]["initial_capital"]
    bh_equity = initial_capital * df_oos["btc_close"] / df_oos["btc_close"].iloc[0]
    bh_return_pct = (bh_equity.iloc[-1] / initial_capital - 1) * 100
    ml_equity = result["equity_curve"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("จำนวนเทรด", summary["n_trades"])
    c2.metric(
        "ผลตอบแทนรวม", f"{summary['total_return_pct']:.1f}%",
        delta=f"{summary['total_return_pct'] - bh_return_pct:+.1f}% เทียบ B&H",
    )
    c3.metric("Win Rate", f"{summary['win_rate_pct']:.1f}%")
    c4.metric("Max Drawdown", f"{summary['max_drawdown_pct']:.1f}%")
    c5.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    c6.metric("Buy & Hold", f"{bh_return_pct:.1f}%")

    if summary["n_trades"] == 0:
        st.warning("⚠️ ไม่มีเทรดเกิดขึ้นเลยที่ threshold นี้ — ลองลด Confidence Threshold ในแถบด้านซ้าย")

    # Equity curve เทียบ Buy & Hold + กราฟ drawdown ด้านล่าง
    ml_dd = (ml_equity / ml_equity.cummax() - 1) * 100
    bh_dd = (bh_equity / bh_equity.cummax() - 1) * 100

    fig_equity = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
    )
    fig_equity.add_trace(go.Scatter(
        x=ml_equity.index, y=ml_equity.values, name="ML Ensemble", line=dict(color="#2ecc71"),
    ), row=1, col=1)
    fig_equity.add_trace(go.Scatter(
        x=bh_equity.index, y=bh_equity.values, name="Buy & Hold",
        line=dict(color="#95a5a6", dash="dash"),
    ), row=1, col=1)
    fig_equity.add_trace(go.Scatter(
        x=ml_dd.index, y=ml_dd.values, name="Drawdown (ML)", fill="tozeroy",
        line=dict(color="#e74c3c", width=1), fillcolor="rgba(231,76,60,0.25)",
    ), row=2, col=1)
    fig_equity.add_trace(go.Scatter(
        x=bh_dd.index, y=bh_dd.values, name="Drawdown (B&H)",
        line=dict(color="#95a5a6", width=1, dash="dot"),
    ), row=2, col=1)
    fig_equity.update_yaxes(title_text="Equity (USD)", row=1, col=1)
    fig_equity.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig_equity.update_layout(
        height=480, margin=dict(t=20, b=20), hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # --- จุด Buy/Sell บนกราฟราคา (เฉพาะช่วง out-of-sample ที่ backtest จริง) ---
    st.subheader("📍 จุดซื้อ-ขายบนกราฟราคา")
    trade_log = result["trade_log"]

    fig_trades = go.Figure()
    fig_trades.add_trace(go.Scatter(
        x=df_oos.index, y=df_oos["btc_close"],
        name="BTC Close", line=dict(color="#f2a900"), opacity=0.6,
    ))

    if not trade_log.empty and "entry_date" in trade_log.columns:
        entries = trade_log[trade_log["entry_date"].notna()]
        if not entries.empty:
            fig_trades.add_trace(go.Scatter(
                x=entries["entry_date"], y=entries["entry_price"],
                mode="markers", name="เข้าซื้อ (Buy)",
                marker=dict(symbol="triangle-up", size=11, color="#2ecc71"),
            ))

    if not trade_log.empty and "exit_date" in trade_log.columns:
        exits = trade_log[trade_log["exit_date"].notna()]
        reason_style = {
            "stop_loss": ("#e74c3c", "Stop Loss"),
            "take_profit": ("#3498db", "Take Profit"),
            "signal": ("#e67e22", "สัญญาณขาย"),
        }
        for reason, (color, label) in reason_style.items():
            subset = exits[exits["reason"] == reason]
            if not subset.empty:
                fig_trades.add_trace(go.Scatter(
                    x=subset["exit_date"], y=subset["exit_price"],
                    mode="markers", name=f"ออก ({label})",
                    marker=dict(symbol="triangle-down", size=11, color=color),
                ))

    fig_trades.update_layout(height=400, margin=dict(t=20, b=20), yaxis_title="ราคา (USD)")
    st.plotly_chart(fig_trades, use_container_width=True)

    st.caption(
        "หมายเหตุ: ผลลัพธ์นี้คำนวณจาก out-of-sample prediction เท่านั้น "
        f"(ครอบคลุม {len(df_oos)} จาก {len(df)} แถว) เพื่อป้องกัน data leakage / in-sample bias"
    )

    # --- Threshold sweep ---
    st.subheader("🎚️ Threshold Sweep")
    sweep_df = run_threshold_sweep(
        df_oos, proba_up,
        tuple(sorted((k, v) for k, v in config["backtest"].items() if k != "confidence_threshold")),
        data_key=f"{len(df_oos)}|{df_oos.index[-1]}|{proba_up.sum():.6f}",
    )

    fig_sweep = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Total Return (%)", "Sharpe Ratio", "จำนวนเทรด"),
    )
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["total_return_pct"], mode="lines+markers",
        line=dict(color="#2ecc71"), showlegend=False,
    ), row=1, col=1)
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["sharpe_ratio"], mode="lines+markers",
        line=dict(color="#3498db"), showlegend=False,
    ), row=2, col=1)
    fig_sweep.add_trace(go.Bar(
        x=sweep_df["threshold"], y=sweep_df["n_trades"], marker_color="#e67e22", showlegend=False,
    ), row=3, col=1)
    # ขีดแนวตั้งที่ threshold ปัจจุบันจาก slider (ขยับตามแบบ real-time)
    fig_sweep.add_vline(x=threshold, line_dash="dash", line_color="#f1c40f")
    fig_sweep.update_xaxes(title_text="Confidence Threshold", row=3, col=1)
    fig_sweep.update_layout(height=520, margin=dict(t=40, b=20))
    st.plotly_chart(fig_sweep, use_container_width=True)

    reliable = sweep_df[sweep_df["n_trades"] >= MIN_TRADES_FOR_BEST]
    if len(reliable) > 0:
        best = reliable.loc[reliable["sharpe_ratio"].idxmax()]
        st.caption(
            f"Sharpe ดีที่สุด (เฉพาะ threshold ที่มีเทรด ≥ {MIN_TRADES_FOR_BEST} ครั้ง): "
            f"**{best['threshold']:.2f}** (Sharpe {best['sharpe_ratio']:.2f}, "
            f"{int(best['n_trades'])} เทรด, ผลตอบแทน {best['total_return_pct']:.1f}%) • "
            "เส้นประสีเหลือง = threshold ที่เลือกอยู่ตอนนี้ • "
            "ระวัง: การเลือก threshold จากชุด OOS เดียวกับที่ใช้ประเมินผล มีโอกาส overfit ต่อช่วงข้อมูลนี้"
        )
    else:
        st.caption(f"ไม่มี threshold ไหนที่มีเทรดถึง {MIN_TRADES_FOR_BEST} ครั้ง — ผลอาจไม่น่าเชื่อถือทางสถิติ")

    # --- Comparison with rule-based ---
    st.subheader("⚖️ เทียบกับ Rule-Based Strategies")
    signals = get_all_strategies(df_oos)
    signals["ML_Ensemble"] = signal
    comparison = compare_strategies(
        df_oos, signals, config["backtest"], confidence_map={"ML_Ensemble": pd.Series(proba_up)}
    )

    # หมายเหตุ: ใช้ HTML table แทน st.dataframe() เพราะ st.dataframe() พึ่งพา
    # pyarrow เบื้องหลัง ซึ่งถูก Application Control policy ของเครื่องนี้บล็อกอยู่
    fmt_map = {
        "total_return_pct": "{:.1f}%", "win_rate_pct": "{:.1f}%",
        "max_drawdown_pct": "{:.1f}%", "sharpe_ratio": "{:.2f}",
        "avg_win": "${:,.0f}", "avg_loss": "${:,.0f}", "final_equity": "${:,.0f}",
        "annual_return_pct": "{:.1f}%", "sortino_ratio": "{:.2f}", "calmar_ratio": "{:.2f}",
        "profit_factor": "{:.2f}", "exposure_pct": "{:.0f}%", "avg_holding_days": "{:.1f}",
    }
    comparison_display = comparison.copy()
    for col, fmt in fmt_map.items():
        if col in comparison_display.columns:
            comparison_display[col] = comparison_display[col].apply(lambda v: fmt.format(v))

    st.markdown(
        comparison_display.to_html(index=True, border=0, classes="comparison-table"),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .comparison-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        .comparison-table th { text-align: left; padding: 6px 10px; border-bottom: 2px solid #444; }
        .comparison-table td { padding: 6px 10px; border-bottom: 1px solid #333; }
        .comparison-table tr:hover { background-color: rgba(255,255,255,0.05); }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("ยังไม่มีโมเดลที่เทรนไว้ — รัน `python -m main train` ก่อน")

# --- Overfitting monitor ---
st.subheader("🔍 Overfitting Monitor (Walk-Forward)")
if WF_RESULTS_PATH.exists():
    wf = pd.read_csv(WF_RESULTS_PATH)
    needed = {"fold", "train_accuracy", "val_accuracy", "accuracy"}
    if needed.issubset(wf.columns):
        fold_labels = [f"Fold {int(f)}" for f in wf["fold"]]
        fig_of = go.Figure()
        fig_of.add_trace(go.Bar(x=fold_labels, y=wf["train_accuracy"] * 100, name="Train", marker_color="#3498db"))
        fig_of.add_trace(go.Bar(x=fold_labels, y=wf["val_accuracy"] * 100, name="Val", marker_color="#9b59b6"))
        fig_of.add_trace(go.Bar(x=fold_labels, y=wf["accuracy"] * 100, name="OOS (Test)", marker_color="#e67e22"))
        fig_of.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% (เดาสุ่ม)")
        fig_of.update_layout(
            barmode="group", height=320, margin=dict(t=20, b=20), yaxis_title="Accuracy (%)",
        )
        st.plotly_chart(fig_of, use_container_width=True)

        gap = (wf["train_accuracy"] - wf["accuracy"]).mean() * 100
        g1, g2, g3 = st.columns(3)
        g1.metric("Train Accuracy เฉลี่ย", f"{wf['train_accuracy'].mean() * 100:.1f}%")
        g2.metric("OOS Accuracy เฉลี่ย", f"{wf['accuracy'].mean() * 100:.1f}%")
        g3.metric("Gap (Train − OOS)", f"{gap:.1f} pp")
        if gap > 15:
            st.warning("⚠️ Gap ระหว่าง Train กับ OOS กว้างมาก — โมเดลน่าจะยัง overfit อยู่")
        st.caption(
            "หมายเหตุ: หลัง tune เสร็จโมเดลถูก refit ด้วย train+val รวมกัน ดังนั้น Val ในกราฟนี้ไม่ใช่ข้อมูลที่โมเดลไม่เคยเห็นจริง "
            "— ให้ดู gap จาก Train เทียบกับ OOS เป็นหลัก"
        )
    else:
        st.info("ไฟล์ walk_forward_results.csv ไม่มีคอลัมน์ train/val accuracy — รัน `python -m main train` ใหม่")
else:
    st.info("ยังไม่มี results/walk_forward_results.csv — รัน `python -m main train` ก่อน")

st.divider()
st.caption("Dashboard นี้อ่านข้อมูลจากไฟล์ที่สร้างโดย main.py เท่านั้น ไม่ได้ดึงข้อมูลสดจากตลาดโดยตรง")