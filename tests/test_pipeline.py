"""
เทสต์ที่ป้องกันบั๊กสำคัญของ pipeline ไม่ให้กลับมา  รัน:  python -m pytest -q
ใช้ข้อมูลจำลองทั้งหมด (ไม่ต้องใช้อินเทอร์เน็ต ไม่แตะไฟล์ใน data/ หรือ models/)
"""
import numpy as np
import pandas as pd
import pytest

from src import data_loader, features
from src.backtester import (
    compare_strategies, compute_dynamic_position_size, run_backtest, signals_from_confidence,
)
from src.evaluation import align_oos
from src.live_signal import get_latest_signal
from src.model import EnsembleModel, walk_forward_splits
from src.utils import load_config

HORIZON = 7


def make_ohlcv(n=500, start_price=30000, seed=0, end="2026-10-20"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=n, freq="D")
    price = start_price * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    df = pd.DataFrame({
        "open": price * (1 + rng.normal(0, 0.003, n)),
        "high": price * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": price * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": price,
        "volume": rng.lognormal(15, 0.4, n),
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture(scope="module")
def config():
    cfg = load_config()
    cfg["model"]["target_horizon"] = HORIZON
    return cfg


@pytest.fixture(scope="module")
def merged():
    raw = {"btc": make_ohlcv(seed=1), "eth": make_ohlcv(start_price=2000, seed=2),
           "dxy": make_ohlcv(start_price=100, seed=3), "vix": make_ohlcv(start_price=18, seed=4)}
    return data_loader.merge_cross_assets(raw)


# ------------------------------------------------------------------ features / live row
def test_live_matrix_reaches_latest_bar_but_training_matrix_does_not(merged, config):
    train = features.build_feature_matrix(merged, config)
    live = features.build_live_matrix(merged, config)
    assert live.index[-1] == merged.index[-1]
    assert train.index[-1] == merged.index[-1 - HORIZON]


def test_live_last_row_has_no_target_and_complete_features(merged, config):
    live = features.build_live_matrix(merged, config)
    cols = features.get_feature_columns(live)
    assert pd.isna(live["target"].iloc[-1])
    assert not live[cols].isna().any().any()


def test_training_target_is_binary_ints_without_fake_zero_labels(merged, config):
    train = features.build_feature_matrix(merged, config)
    assert set(train["target"].unique()) <= {0, 1}
    # target ของแถวสุดท้ายต้องตรงกับราคาจริง HORIZON วันถัดไป (ไม่ใช่ 0 ปลอมจาก NaN)
    last = train.index[-1]
    pos = merged.index.get_loc(last)
    expected = int(merged["btc_close"].iloc[pos + HORIZON] > merged["btc_close"].iloc[pos])
    assert train.loc[last, "target"] == expected


def test_price_level_columns_are_not_model_features(merged, config):
    cols = set(features.get_feature_columns(features.build_live_matrix(merged, config)))
    for banned in ("bb_upper", "bb_mid", "bb_lower", "atr", "macd", "macd_signal", "macd_hist", "btc_close"):
        assert banned not in cols
    assert {"bb_width", "macd_pct", "atr_pct", "rsi"} <= cols


def test_rsi_is_100_when_price_only_rises():
    s = pd.Series(np.arange(1.0, 60.0))
    assert features._rsi(s, 14).iloc[-1] == pytest.approx(100.0)


# ------------------------------------------------------------------ walk-forward purge
@pytest.mark.parametrize("purge", [0, 7, 14])
def test_walk_forward_leaves_purge_gap_and_never_overlaps(purge):
    splits = walk_forward_splits(1000, 5, 0.15, purge=purge)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert train_idx.max() + purge < test_idx.min()      # label ของแถวท้าย train ไม่ล้ำเข้า test
        assert len(set(train_idx) & set(test_idx)) == 0


def test_with_purge_last_train_label_window_ends_before_test():
    train_idx, test_idx = walk_forward_splits(1000, 5, 0.15, purge=HORIZON)[0]
    assert train_idx.max() + HORIZON < test_idx.min()


# ------------------------------------------------------------------ backtester
def _bt_frame(closes, lows=None, highs=None):
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "btc_close": closes, "btc_low": lows if lows is not None else closes,
        "btc_high": highs if highs is not None else closes,
    }, index=idx)


def _cfg(**over):
    base = dict(initial_capital=10000, fee_pct=0.0, slippage_pct=0.0, stop_loss_pct=0.05, take_profit_pct=0.10,
                position_size_pct=1.0, position_sizing_mode="fixed", confidence_threshold=0.55)
    base.update(over)
    return base


def test_stop_loss_exits_at_stop_price_not_at_close():
    # วันที่ 2 low ทะลุ -5% แต่ปิดที่ -12% -> ต้องออกที่ -5% เป๊ะ
    df = _bt_frame([100, 100, 88], lows=[100, 100, 85], highs=[100, 100, 100])
    sig = pd.Series([0, 1, 0])
    res = run_backtest(df, sig, _cfg())
    t = res["trade_log"].dropna(subset=["pnl"]).iloc[0]
    assert t["reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(100 * 0.95)


def test_same_day_stop_and_take_profit_prefers_stop():
    df = _bt_frame([100, 100, 100], lows=[100, 100, 90], highs=[100, 100, 120])
    res = run_backtest(df, pd.Series([0, 1, 0]), _cfg())
    assert res["trade_log"].dropna(subset=["pnl"]).iloc[0]["reason"] == "stop_loss"


def test_open_position_is_closed_at_end_of_data_and_counted():
    df = _bt_frame([100, 100, 105, 110])
    res = run_backtest(df, pd.Series([0, 1, 0, 0]), _cfg(take_profit_pct=0.5))
    exits = res["trade_log"].dropna(subset=["pnl"])
    assert list(exits["reason"]) == ["end_of_data"]
    assert res["summary"]["n_trades"] == 1
    assert res["summary"]["final_equity"] == pytest.approx(11000.0)


def test_fees_and_slippage_reduce_final_equity():
    df = _bt_frame([100, 100, 110])
    sig = pd.Series([0, 1, -1])
    free = run_backtest(df, sig, _cfg())["summary"]["final_equity"]
    paid = run_backtest(df, sig, _cfg(fee_pct=0.001, slippage_pct=0.0005))["summary"]["final_equity"]
    assert paid < free


def test_no_signals_means_flat_equity_and_zero_trades():
    df = _bt_frame([100, 101, 102, 103])
    res = run_backtest(df, pd.Series([0, 0, 0, 0]), _cfg())
    assert res["summary"]["n_trades"] == 0
    assert res["summary"]["final_equity"] == 10000


def test_compare_strategies_includes_buy_and_hold_that_tracks_price():
    df = _bt_frame([100, 100, 150])
    out = compare_strategies(df, {"X": pd.Series([0, 0, 0])}, _cfg())
    assert "Buy_and_Hold" in out.index
    assert out.loc["Buy_and_Hold", "total_return_pct"] == pytest.approx(50.0)
    assert out.loc["X", "total_return_pct"] == 0


def test_signals_threshold_and_gap():
    proba = np.array([0.80, 0.56, 0.54, 0.50, 0.46, 0.44, 0.10])
    s = signals_from_confidence(proba, {"confidence_threshold": 0.55, "min_confidence_gap": 0.05})
    assert list(s) == [1, 1, 0, 0, 0, -1, -1]
    # threshold ต่ำกว่า 0.5+gap -> gap เริ่มมีผล: 0.52 ผ่าน threshold 0.51 แต่ไม่ผ่าน gap 0.05
    s2 = signals_from_confidence(np.array([0.52, 0.56]), {"confidence_threshold": 0.51, "min_confidence_gap": 0.05})
    assert list(s2) == [0, 1]


def test_dynamic_size_scales_between_min_and_max():
    assert compute_dynamic_position_size(0.55, 0.55, 0.3, 0.8) == pytest.approx(0.3)
    assert compute_dynamic_position_size(1.0, 0.55, 0.3, 0.8) == pytest.approx(0.8)
    assert 0.3 < compute_dynamic_position_size(0.8, 0.55, 0.3, 0.8) < 0.8


# ------------------------------------------------------------------ data loader / OOS alignment
def test_drop_incomplete_last_bar_removes_today_only():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=[today - pd.Timedelta(days=2), today - pd.Timedelta(days=1), today])
    out = data_loader.drop_incomplete_last_bar(df)
    assert out.index.max() == today - pd.Timedelta(days=1)


def test_stale_cache_detection():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    fresh = pd.DataFrame({"close": [1.0]}, index=[today - pd.Timedelta(days=1)])
    old = pd.DataFrame({"close": [1.0]}, index=[today - pd.Timedelta(days=5)])
    assert not data_loader._is_stale(fresh, "btc")
    assert data_loader._is_stale(old, "btc")
    assert data_loader._is_stale(old, "vix")       # 5 วันเกินเกณฑ์ 4 วัน
    assert data_loader._is_stale(None, "btc")


def test_oos_alignment_survives_longer_feature_matrix():
    idx = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({"x": range(100)}, index=idx)
    oos = pd.Series(np.nan, index=idx[:90])        # oos_proba เก่า สั้นกว่า matrix ใหม่
    oos.iloc[40:90] = np.linspace(0.3, 0.7, 50)
    d, p = align_oos(df, oos)
    assert len(d) == len(p) == 50 and d.index[0] == idx[40]


# ------------------------------------------------------------------ live signal end-to-end
def test_latest_signal_is_for_latest_bar_and_uses_models_feature_list(merged, config):
    cfg = {**config, "model": {**config["model"], "algorithms": ["randomforest"]}}
    cfg["model"]["optuna"] = {**config["model"]["optuna"], "enabled": False}
    train = features.build_feature_matrix(merged, cfg)
    cols = features.get_feature_columns(train)
    split = int(len(train) * 0.8)
    model = EnsembleModel(["randomforest"]).fit(
        train[cols].iloc[:split], train["target"].iloc[:split],
        train[cols].iloc[split:], train["target"].iloc[split:], cfg["model"]["optuna"])
    assert model.feature_cols == cols

    live = features.build_live_matrix(merged, cfg)
    sig = get_latest_signal(live, model, cfg)
    assert sig["as_of_date"] == merged.index[-1]
    assert 0.0 <= sig["proba_up"] <= 1.0
    assert sig["signal_label"] in {"BUY", "SELL", "HOLD"}
    assert sig["current_price"] == pytest.approx(merged["btc_close"].iloc[-1])

    # ฟีเจอร์หายไปจากข้อมูล -> ต้อง error ชัดเจน ไม่ predict มั่ว
    with pytest.raises(ValueError, match="train"):
        get_latest_signal(live.drop(columns=[cols[0]]), model, cfg)


# ------------------------------------------------------------------ load_all_data: cache / force refresh
def _write_btc_cache(tmp_path, last_age_days):
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    idx = pd.date_range(end=today - pd.Timedelta(days=last_age_days), periods=30, freq="D", name="date")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    df.to_csv(tmp_path / "BTC_USD.csv")
    return df


@pytest.fixture
def loader_cfg(tmp_path, config, monkeypatch):
    cfg = {**config, "data": {**config["data"], "cross_assets": {}}, "paths": {**config["paths"], "data_dir": str(tmp_path)}}
    monkeypatch.setattr(data_loader, "fetch_onchain_metrics", lambda start: pd.DataFrame())
    return cfg


def test_fresh_cache_is_used_without_downloading(tmp_path, loader_cfg, monkeypatch):
    _write_btc_cache(tmp_path, last_age_days=1)
    monkeypatch.setattr(data_loader, "_download_symbol", lambda *a, **k: pytest.fail("ไม่ควรดึงใหม่เมื่อ cache ยังสด"))
    assert "btc" in data_loader.load_all_data(loader_cfg)


def test_stale_cache_is_refreshed_and_saved(tmp_path, loader_cfg, monkeypatch):
    _write_btc_cache(tmp_path, last_age_days=6)
    fresh = make_ohlcv(n=40, end=(pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    monkeypatch.setattr(data_loader, "_download_symbol", lambda *a, **k: fresh)
    out = data_loader.load_all_data(loader_cfg)
    assert out["btc"].index.max() == fresh.index.max()


def test_stale_cache_falls_back_when_download_fails(tmp_path, loader_cfg, monkeypatch):
    _write_btc_cache(tmp_path, last_age_days=6)
    monkeypatch.setattr(data_loader, "_download_symbol", lambda *a, **k: None)
    assert "btc" in data_loader.load_all_data(loader_cfg)          # ใช้ cache เดิมพร้อม warning


def test_force_refresh_fails_loudly_if_btc_cannot_be_downloaded(tmp_path, loader_cfg, monkeypatch):
    _write_btc_cache(tmp_path, last_age_days=1)
    monkeypatch.setattr(data_loader, "_download_symbol", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="BTC"):
        data_loader.load_all_data(loader_cfg, force_refresh=True)
