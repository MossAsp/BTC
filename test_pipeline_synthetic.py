"""
test_pipeline_synthetic.py — ทดสอบทั้ง pipeline ด้วยข้อมูลจำลอง (สุ่มแบบ random walk)
ใช้ตรวจว่าโค้ดทุกส่วนรันได้จริงโดยไม่ error ก่อนที่จะให้ผู้ใช้รันกับข้อมูลจริง
(สคริปต์นี้ไม่ใช่ส่วนของ pipeline หลัก ใช้สำหรับ debug เท่านั้น)
"""
import numpy as np
import pandas as pd

from src import features
from src.backtester import compare_strategies, run_backtest, signals_from_confidence
from src.model import EnsembleModel, run_walk_forward_training
from src.rule_strategies import get_all_strategies
from src.utils import PROJECT_ROOT, load_config

np.random.seed(42)


def make_synthetic_ohlcv(n=800, start_price=30000, symbol="TEST"):
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    returns = np.random.normal(0.0005, 0.02, n)
    # แทรก trend ช่วงกลางเพื่อให้มี regime bull/bear ชัดเจนสำหรับทดสอบ
    returns[200:350] += 0.01
    returns[500:600] -= 0.008
    price = start_price * np.cumprod(1 + returns)

    high = price * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = price * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(15, 0.5, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": price, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


print("=== 1. สร้างข้อมูลจำลอง ===")
btc = make_synthetic_ohlcv(800, 30000)
eth = make_synthetic_ohlcv(800, 2000)
dxy = make_synthetic_ohlcv(800, 100)
vix = make_synthetic_ohlcv(800, 18)
raw = {"btc": btc, "eth": eth, "dxy": dxy, "vix": vix}
print(f"BTC rows: {len(btc)}")

print("\n=== 2. merge_cross_assets ===")
from src.data_loader import merge_cross_assets
merged = merge_cross_assets(raw)
print(merged.columns.tolist())
assert not merged.empty

print("\n=== 3. build_feature_matrix ===")
config = load_config()
df = features.build_feature_matrix(merged, config)
print(f"หลังสร้างฟีเจอร์: {df.shape}")
feature_cols = features.get_feature_columns(df)
print(f"จำนวนฟีเจอร์: {len(feature_cols)}")
assert df["target"].isin([0, 1]).all()
live = features.build_live_matrix(merged, config)
horizon = config["model"]["target_horizon"]
assert live.index[-1] == merged.index[-1], "live matrix ต้องยาวถึงแท่งล่าสุดจริง"
assert df.index[-1] < live.index[-1], "เมทริกซ์เทรนต้องสั้นกว่า (ตัดแถวที่ไม่มี target)"
assert live["target"].iloc[-1] != live["target"].iloc[-1], "แถวล่าสุดต้องยังไม่มี target (NaN)"
print(f"live matrix: {len(live)} แถว ถึง {live.index[-1].date()} | train matrix ถึง {df.index[-1].date()} (horizon={horizon})")
assert set(df["regime"].unique()).issubset({-1, 0, 1})
print("regime distribution:\n", df["regime"].value_counts())

print("\n=== 4. rule-based strategies ===")
signals = get_all_strategies(df)
for name, sig in signals.items():
    print(f"{name}: buy={sum(sig==1)} sell={sum(sig==-1)} hold={sum(sig==0)}")

print("\n=== 5. backtest rule-based ===")
comparison = compare_strategies(df, signals, config["backtest"])
print(comparison)

print("\n=== 6. train ensemble (ลด optuna trial ให้เร็วสำหรับทดสอบ) ===")
config["model"]["optuna"]["n_trials"] = 3
config["model"]["walk_forward"]["n_splits"] = 3
model, fold_results, oos_proba = run_walk_forward_training(df, feature_cols, config)
print(fold_results)
print(f"OOS coverage: {oos_proba.notna().sum()} / {len(df)} แถว")

print("\n=== 7. backtest ML ensemble (out-of-sample เท่านั้น) ===")
valid_mask = oos_proba.notna()
df_oos = df.loc[valid_mask]
proba_up = oos_proba.loc[valid_mask].values
print(f"proba_up distribution: min={proba_up.min():.3f} max={proba_up.max():.3f} mean={proba_up.mean():.3f}")
sig_ml = signals_from_confidence(proba_up, config["backtest"])
print(f"ML signal counts: {sig_ml.value_counts().to_dict()}")
result = run_backtest(df_oos, sig_ml, config["backtest"])
print(result["summary"])
assert result["summary"]["total_return_pct"] < 10000, "total return ผิดปกติ ยังมีบั๊กเรื่อง compounding"

print("\n=== 8. save/load model ===")
test_model_path = PROJECT_ROOT / "models" / "test_model.joblib"
model.save(test_model_path)
loaded = EnsembleModel.load(test_model_path)
proba2 = loaded.predict_proba(df[feature_cols][:5])
print("Loaded model proba sample:", proba2[:2])

print("\n=== ALL TESTS PASSED ===")
