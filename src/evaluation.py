"""
evaluation.py — ตัวช่วยโหลด out-of-sample prediction ให้ตรงแถวกับ feature matrix

ปัญหาเดิม: สคริปต์ต่างๆ ใช้ `df.loc[oos.notna()]` ซึ่ง boolean Series ต้องมี index ตรงกับ df เป๊ะ
พอ `fetch-data` รันใหม่ (เช่นจาก run_daily_alert.ps1) feature matrix ยาวขึ้นแต่ oos_proba.csv ยังเป็นของเดิม
index เลยไม่ตรงกัน -> pandas โยน IndexingError (dashboard/backtest/sweep พังหลังรัน daily alert)
ที่นี่จับคู่ด้วย index วันที่ (intersection) แทน จึงไม่พังไม่ว่า matrix จะยาวขึ้นแค่ไหน
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("evaluation")


def load_oos(df: pd.DataFrame, oos_path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """คืน (df_oos, proba_up) ที่แถวตรงกันตามวันที่ โดยใช้เฉพาะวันที่มี out-of-sample prediction"""
    oos = pd.read_csv(oos_path, index_col="date", parse_dates=True)["proba_up"].dropna()
    return align_oos(df, oos)


def align_oos(df: pd.DataFrame, oos: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    oos = oos.dropna()
    common = df.index.intersection(oos.index)
    if len(common) < len(oos):
        logger.warning(
            f"oos_proba มี {len(oos)} แถว แต่หาแถวตรงกันใน feature matrix ได้ {len(common)} แถว "
            f"— ถ้าเพิ่งเปลี่ยน config/ฟีเจอร์ ควรรัน `python -m main train` ใหม่"
        )
    if len(common) == 0:
        raise ValueError("ไม่มีวันที่ตรงกันระหว่าง feature matrix กับ oos_proba.csv — รัน train ใหม่")
    return df.loc[common], oos.loc[common].to_numpy()


def segment_slices(n: int, k: int) -> list[np.ndarray]:
    """แบ่ง index 0..n-1 เป็น k ช่วงเรียงตามเวลา (ขนาดเท่ากันที่สุด)"""
    return [seg for seg in np.array_split(np.arange(n), k) if len(seg) > 0]
