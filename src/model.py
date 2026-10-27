"""
model.py — Ensemble model (XGBoost + LightGBM + RandomForest) พร้อม
  - Walk-Forward Validation (กันการ leak ข้อมูลอนาคต ซึ่งสำคัญมากกับ time series)
  - Optuna hyperparameter tuning
  - Soft-voting ensemble (เฉลี่ยความน่าจะเป็นจากทุกโมเดล ไม่ใช่แค่ vote เสียงข้างมาก)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.utils import PROJECT_ROOT, get_logger

logger = get_logger("model")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # กัน log ของ optuna รกจอ


def walk_forward_splits(n_samples: int, n_splits: int, test_size_ratio: float, purge: int = 0):
    """
    สร้าง index สำหรับ Walk-Forward Validation (expanding window):
    แต่ละ fold เทรนด้วยข้อมูลทั้งหมดถึงจุดนั้น แล้วทดสอบกับช่วงถัดไปที่ไม่เคยเห็น

    purge: เว้นช่องว่างระหว่างท้าย train กับต้น test อย่างน้อย target_horizon แถว
      เหตุผล: label ของแถวท้าย train (ทำนายอีก N วันข้างหน้า) ใช้ราคาที่ล้ำเข้าไปในช่วง test
      ถ้าไม่เว้น โมเดลจะ "เห็น" ราคาอนาคตของช่วง test ผ่าน label -> OOS ดีเกินจริงเล็กน้อย
      (ดู López de Prado, Advances in Financial ML, บทที่ 7: purging & embargo)
    """
    test_size = int(n_samples * test_size_ratio)
    splits = []
    for i in range(n_splits):
        test_end = n_samples - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start - purge
        if train_end <= 0:
            continue
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def _objective(trial: optuna.Trial, model_name: str, X_train, y_train, X_val, y_val):
    """Objective function ของ Optuna แยกตามชนิดโมเดล"""
    if model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 5),  # ลด upper bound จาก 8 เหลือ 5 กัน tree ลึกเกินไป
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),  # เพดานลดจาก 1.0 บังคับให้สุ่มข้อมูลเสมอ
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),  # เพิ่ม: กัน leaf เล็กเกินไป
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),  # เพิ่ม: L1 regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),  # เพิ่ม: L2 regularization
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params, random_state=42, n_jobs=-1)
    elif model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),  # ลด upper bound จาก 63 เหลือ 31
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),  # เพิ่ม: กัน leaf เล็กเกินไป
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),  # เพิ่ม: L1 regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),  # เพิ่ม: L2 regularization
        }
        model = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    elif model_name == "randomforest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 6),  # ลด upper bound จาก 15 เหลือ 6
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 50),  # เพิ่ม lower bound จาก 2 เป็น 15
            "max_features": trial.suggest_float("max_features", 0.4, 0.8),  # เพิ่ม: จำกัด feature ต่อ split
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"ไม่รู้จักโมเดล: {model_name}")

    model.fit(X_train, y_train)

    # ประเมินทั้ง train และ val แล้วลงโทษ (penalize) ถ้า gap ระหว่างสองค่ากว้างเกินไป
    # ป้องกัน Optuna เลือกพารามิเตอร์ที่ overfit (train ดีมากแต่ val ไม่ดี) เหมือนที่เจอปัญหามาก่อน
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_f1 = f1_score(y_train, train_preds, zero_division=0)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)

    overfit_gap = max(0.0, train_f1 - val_f1)
    penalty = overfit_gap * 0.5  # ยิ่ง gap เยอะ ยิ่งหักคะแนนเยอะ
    return val_f1 - penalty


def tune_model(model_name: str, X_train, y_train, X_val, y_val, n_trials: int, timeout: int) -> dict:
    """หา hyperparameter ที่ดีที่สุดด้วย Optuna สำหรับโมเดลหนึ่งตัว"""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),  # fix seed เพื่อให้ hyperparameter search reproducible
    )
    study.optimize(
        lambda trial: _objective(trial, model_name, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    logger.info(f"[{model_name}] best F1={study.best_value:.4f} params={study.best_params}")
    return study.best_params


def build_model(model_name: str, params: dict):
    """สร้าง instance ของโมเดลจากชื่อและ hyperparameter ที่ tune แล้ว"""
    if model_name == "xgboost":
        return XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric="logloss")
    if model_name == "lightgbm":
        return LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    if model_name == "randomforest":
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    raise ValueError(f"ไม่รู้จักโมเดล: {model_name}")


class EnsembleModel:
    """
    Ensemble แบบ soft-voting: เทรนโมเดลย่อยหลายตัวแยกกัน
    แล้วตอน predict เฉลี่ยความน่าจะเป็น (predict_proba) จากทุกตัว
    ข้อดีเทียบกับ hard voting: ได้ค่า confidence ที่ใช้กำหนด threshold ตอน backtest ได้ตรงไปตรงมา
    """

    def __init__(self, model_names: list[str]):
        self.model_names = model_names
        self.models: dict[str, object] = {}
        self.best_params: dict[str, dict] = {}
        self.feature_cols: list[str] | None = None  # จำชื่อฟีเจอร์ตอนเทรน ใช้กัน feature ไม่ตรงตอน predict

    def fit(self, X_train, y_train, X_val, y_val, optuna_cfg: dict):
        if hasattr(X_train, "columns"):
            self.feature_cols = list(X_train.columns)
        for name in self.model_names:
            if optuna_cfg.get("enabled", True):
                params = tune_model(
                    name, X_train, y_train, X_val, y_val,
                    n_trials=optuna_cfg["n_trials"], timeout=optuna_cfg["timeout_sec"],
                )
            else:
                params = {}
            model = build_model(name, params)
            # เทรนด้วย train+val รวมกันหลัง tune เสร็จ เพื่อใช้ข้อมูลให้คุ้มค่าที่สุด
            X_full = pd.concat([X_train, X_val])
            y_full = pd.concat([y_train, y_val])
            model.fit(X_full, y_full)
            self.models[name] = model
            self.best_params[name] = params
        return self

    def predict_proba(self, X) -> np.ndarray:
        """เฉลี่ย predict_proba จากทุกโมเดลย่อย -> คืน array shape (n_samples, 2)"""
        probs = [model.predict_proba(X) for model in self.models.values()]
        return np.mean(probs, axis=0)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"บันทึกโมเดลไปที่ {path}")

    @staticmethod
    def load(path: str | Path) -> "EnsembleModel":
        return joblib.load(path)


def evaluate(y_true, y_pred, y_proba=None) -> dict:
    """คำนวณ metric มาตรฐานสำหรับ classification"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        # log_loss ต้อง clip ค่า proba ไม่ให้ชิด 0/1 พอดี ป้องกัน log(0) error
        proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
        metrics["loss"] = log_loss(y_true, proba_clipped)
    return metrics


def run_walk_forward_training(
    df: pd.DataFrame, feature_cols: list[str], config: dict
) -> tuple[EnsembleModel, pd.DataFrame, pd.Series]:
    """
    รัน Walk-Forward Validation เต็มรูปแบบ (เทรน/ทดสอบทีละ fold ตามลำดับเวลา) แล้วเทรนโมเดลสุดท้าย
    ด้วยข้อมูลทั้งหมดสำหรับใช้ predict ข้อมูลใหม่

    *** คืน oos_proba (out-of-sample probability) ด้วย — ใช้ค่านี้ backtest เท่านั้น ***
    ห้ามเอา final_model ไป predict บนข้อมูลที่มันเทรนมา (in-sample, ผลดีเกินจริง)
    oos_proba ของแต่ละแถวมาจากโมเดลที่ "ไม่เคยเห็น" แถวนั้นตอนเทรน

    purge (config model.walk_forward.purge_days, อย่างน้อย = target_horizon) ถูกใช้ 3 ที่:
      1) ระหว่างท้าย train fold กับต้น test      2) ระหว่าง train ย่อยกับ val ที่ใช้ tune
      3) ระหว่าง train กับ val ของโมเดลสุดท้าย
    """
    model_cfg = config["model"]
    wf_cfg = model_cfg["walk_forward"]
    horizon = int(model_cfg["target_horizon"])
    purge = wf_cfg.get("purge_days")
    purge = horizon if purge is None else int(purge)
    if purge < horizon:
        logger.warning(f"purge_days={purge} น้อยกว่า target_horizon={horizon} -> ปรับเป็น {horizon} เพื่อกัน label leakage")
        purge = horizon

    X = df[feature_cols]
    y = df["target"]

    splits = walk_forward_splits(len(df), wf_cfg["n_splits"], wf_cfg["test_size_ratio"], purge=purge)
    fold_results = []
    oos_proba = pd.Series(np.nan, index=df.index)

    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        # แบ่ง train fold เป็น train ย่อย/val (80/20) สำหรับ Optuna โดยเว้น purge ระหว่างสองส่วน
        split_point = int(len(train_idx) * 0.8)
        tr_idx = train_idx[: max(split_point - purge, 1)]
        val_idx = train_idx[split_point:]

        X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        ensemble = EnsembleModel(model_cfg["algorithms"])
        ensemble.fit(X_train, y_train, X_val, y_val, model_cfg["optuna"])

        # train metrics (ไว้ดูว่า overfit ไหม)
        train_proba = ensemble.predict_proba(X_train)[:, 1]
        train_metrics = evaluate(y_train, (train_proba >= 0.5).astype(int), train_proba)

        # val metrics (ใช้ตอน tuning)
        val_proba = ensemble.predict_proba(X_val)[:, 1]
        val_metrics = evaluate(y_val, (val_proba >= 0.5).astype(int), val_proba)

        # test metrics (out-of-sample จริง — ค่านี้คือที่ backtest ใช้)
        proba = ensemble.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, (proba >= 0.5).astype(int), proba)

        metrics["fold"] = fold_i
        metrics["train_loss"] = train_metrics["loss"]
        metrics["train_accuracy"] = train_metrics["accuracy"]
        metrics["val_loss"] = val_metrics["loss"]
        metrics["val_accuracy"] = val_metrics["accuracy"]
        metrics["n_train"] = len(train_idx)
        metrics["n_test"] = len(test_idx)
        metrics["train_end"] = str(df.index[train_idx[-1]])[:10]
        metrics["test_start"] = str(df.index[test_idx[0]])[:10]
        metrics["test_end"] = str(df.index[test_idx[-1]])[:10]
        fold_results.append(metrics)
        logger.info(
            f"Fold {fold_i}/{len(splits)} — "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"test(oos): {metrics}"
        )

        oos_proba.iloc[test_idx] = proba

    results_df = pd.DataFrame(fold_results)
    logger.info(f"ค่าเฉลี่ยทุก fold:\n{results_df.mean(numeric_only=True)}")

    n_oos = oos_proba.notna().sum()
    logger.info(
        f"ได้ out-of-sample prediction ครอบคลุม {n_oos} แถว "
        f"({n_oos / len(df) * 100:.1f}% ของข้อมูลทั้งหมด) — backtest ควรใช้เฉพาะช่วงนี้"
    )

    # โมเดลสุดท้ายสำหรับใช้งานจริงกับข้อมูลใหม่ (ห้ามใช้ backtest กับข้อมูลที่มันเทรน)
    split_point = int(len(X) * 0.8)
    final_model = EnsembleModel(model_cfg["algorithms"])
    cut = max(split_point - purge, 1)
    final_model.fit(
        X.iloc[:cut], y.iloc[:cut],
        X.iloc[split_point:], y.iloc[split_point:],
        model_cfg["optuna"],
    )

    return final_model, results_df, oos_proba
