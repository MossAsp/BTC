"""
main.py — CLI หลักของ pipeline ใช้ Typer
รันคำสั่งทีละขั้น หรือ `all` เพื่อรันทั้งหมดรวดเดียว

  python -m main fetch-data [--force-refresh]   ดึงข้อมูล + สร้าง feature matrix (เทรน) และ live matrix (สัญญาณล่าสุด)
  python -m main train                          เทรน ensemble ด้วย Walk-Forward (มี purge) + Optuna
  python -m main backtest                       backtest ML บน out-of-sample เท่านั้น
  python -m main compare                        เทียบ ML กับ rule-based และ Buy & Hold
  python -m main report                         สร้างตาราง/กราฟ/สรุปผลสำหรับใส่เล่มที่ results/report/
  python -m main all                            fetch -> train -> backtest -> compare -> report
"""
from __future__ import annotations

import pandas as pd
import typer

from src import data_loader, features
from src.backtester import compare_strategies, run_backtest, signals_from_confidence
from src.evaluation import load_oos
from src.model import run_walk_forward_training
from src.paths import FEATURE_DATA_PATH, LIVE_FEATURE_PATH, MODEL_PATH, OOS_PROBA_PATH
from src.rule_strategies import get_all_strategies
from src.utils import PROJECT_ROOT, ensure_dirs, get_logger, load_config

app = typer.Typer(help="BTC/USDT Trading Pipeline CLI")
logger = get_logger("main")


def _prepare_features(config: dict, force_refresh: bool = False) -> pd.DataFrame:
    raw = data_loader.load_all_data(config, force_refresh=force_refresh)
    merged = data_loader.merge_cross_assets(raw)

    train_df = features.build_feature_matrix(merged, config)   # มี target ครบ -> เทรน/backtest
    train_df.to_csv(FEATURE_DATA_PATH)

    live_df = features.build_live_matrix(merged, config)       # ยาวถึงแท่งล่าสุด -> สัญญาณล่าสุด
    live_df.to_csv(LIVE_FEATURE_PATH)
    return train_df


@app.command()
def fetch_data(force_refresh: bool = False):
    """ดึงข้อมูล BTC + cross-asset + on-chain แล้วสร้าง feature matrix"""
    config = load_config()
    ensure_dirs(config)
    df = _prepare_features(config, force_refresh=force_refresh)
    live = pd.read_csv(LIVE_FEATURE_PATH, index_col="date", parse_dates=True)
    typer.echo(
        f"เตรียมข้อมูลเสร็จ: เทรน {len(df)} แถว ({df.index[0].date()} → {df.index[-1].date()}), "
        f"{len(features.get_feature_columns(df))} ฟีเจอร์ | "
        f"แท่งล่าสุดสำหรับสัญญาณ: {live.index[-1].date()}"
    )


@app.command()
def train():
    """เทรนโมเดล ensemble ด้วย Walk-Forward Validation (มี purge) + Optuna tuning"""
    config = load_config()
    ensure_dirs(config)

    if not FEATURE_DATA_PATH.exists():
        typer.echo("ยังไม่มี feature matrix, กำลังดึงข้อมูลใหม่...")
        df = _prepare_features(config)
    else:
        df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)

    feature_cols = features.get_feature_columns(df)
    model, fold_results, oos_proba = run_walk_forward_training(df, feature_cols, config)
    model.save(MODEL_PATH)
    oos_proba.to_csv(OOS_PROBA_PATH, header=["proba_up"])

    results_path = PROJECT_ROOT / config["paths"]["results_dir"] / "walk_forward_results.csv"
    fold_results.to_csv(results_path, index=False)
    typer.echo(f"เทรนเสร็จแล้ว บันทึกโมเดลที่ {MODEL_PATH}")
    typer.echo(fold_results.to_string())
    typer.echo(
        f"บันทึก out-of-sample prediction ที่ {OOS_PROBA_PATH} "
        f"({oos_proba.notna().sum()} แถว) — ใช้ค่านี้ตอน backtest เพื่อความแม่นยำที่แท้จริง"
    )


@app.command()
def backtest():
    """
    backtest โมเดล ML โดยใช้เฉพาะ out-of-sample prediction (oos_proba.csv)
    *** ห้ามใช้ final_model.predict_proba() กับข้อมูลทั้งหมดมา backtest — in-sample ผลดีเกินจริง ***
    """
    config = load_config()
    if not OOS_PROBA_PATH.exists():
        typer.echo("ยังไม่มี out-of-sample prediction — รัน `python -m main train` ก่อน")
        raise typer.Exit(code=1)

    df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)
    df_oos, proba_up = load_oos(df, OOS_PROBA_PATH)   # จับคู่ด้วยวันที่

    signal = signals_from_confidence(proba_up, config["backtest"])
    result = run_backtest(df_oos, signal, config["backtest"], confidence=pd.Series(proba_up))
    typer.echo(f"ผลลัพธ์ ML Ensemble (out-of-sample เท่านั้น, {len(df_oos)} แถว, "
               f"{df_oos.index[0].date()} → {df_oos.index[-1].date()}):")
    for k, v in result["summary"].items():
        typer.echo(f"  {k}: {v}")

    out = PROJECT_ROOT / config["paths"]["results_dir"]
    pd.DataFrame([result["summary"]]).to_csv(out / "ml_backtest_summary.csv", index=False)
    result["trade_log"].to_csv(out / "ml_trade_log.csv", index=False)
    result["equity_curve"].rename("equity").to_csv(out / "ml_equity_curve.csv", index_label="date")


@app.command()
def compare():
    """เทียบ ML ensemble กับ rule-based strategies และ Buy & Hold บนช่วง out-of-sample เดียวกัน"""
    config = load_config()
    df = pd.read_csv(FEATURE_DATA_PATH, index_col="date", parse_dates=True)

    confidence_map = {}
    if OOS_PROBA_PATH.exists():
        df, proba_up = load_oos(df, OOS_PROBA_PATH)
    else:
        typer.echo("ยังไม่มีโมเดลที่เทรนไว้ — เทียบเฉพาะ rule-based strategy บนข้อมูลทั้งหมด")
        proba_up = None

    signals = get_all_strategies(df)
    if proba_up is not None:
        signals["ML_Ensemble"] = signals_from_confidence(proba_up, config["backtest"])
        confidence_map["ML_Ensemble"] = pd.Series(proba_up)

    comparison = compare_strategies(df, signals, config["backtest"], confidence_map=confidence_map)
    typer.echo(comparison.round(3).to_string())

    results_path = PROJECT_ROOT / config["paths"]["results_dir"] / "strategy_comparison.csv"
    comparison.to_csv(results_path)
    typer.echo(f"\nบันทึกผลเปรียบเทียบที่ {results_path}")


@app.command()
def report():
    """สร้างตาราง กราฟ และสรุปผล (results/report/) สำหรับใส่ในเล่มสหกิจ"""
    import generate_report
    generate_report.main()


@app.command(name="all")
def run_all(force_refresh: bool = False):
    """รันทุกขั้นตอนรวดเดียว: fetch -> train -> backtest -> compare -> report"""
    fetch_data(force_refresh=force_refresh)
    train()
    backtest()
    compare()
    report()


if __name__ == "__main__":
    app()
