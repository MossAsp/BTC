"""
utils.py — ฟังก์ชันช่วยเหลือทั่วไป: โหลด config, ตั้งค่า logging
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "config/plan.yaml") -> dict:
    """โหลดไฟล์ config yaml และคืนค่าเป็น dict"""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ config ที่ {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """สร้าง logger ที่เขียนทั้งไปที่ console และไฟล์ log"""
    log_path = Path(log_dir)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:  # ป้องกัน handler ซ้ำเวลาเรียกซ้ำ
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def ensure_dirs(config: dict) -> None:
    """สร้างโฟลเดอร์ที่จำเป็นทั้งหมดตาม config['paths']"""
    for key, rel_path in config.get("paths", {}).items():
        p = PROJECT_ROOT / rel_path
        p.mkdir(parents=True, exist_ok=True)
