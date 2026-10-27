"""
discord_alert.py -- ส่งแจ้งเตือนสัญญาณเทรด BTC/USDT ล่าสุดไปที่ Discord ผ่าน Webhook
อ่านข้อมูลล่าสุดจาก data/live_features.csv + โมเดลที่เทรนไว้ (models/ensemble_model.joblib)

*** โมเดลนี้ทำนาย "ทิศทาง" (ขึ้น/ลง) ใน target_horizon วันข้างหน้า ไม่ใช่ราคาที่แน่นอน ***
*** จึงไม่แสดงราคาในอนาคต เพื่อไม่ให้ดูน่าเชื่อถือเกินความสามารถจริงของโมเดล ***

วิธีรัน (PowerShell):
    python discord_alert.py              # ส่งจริง
    python discord_alert.py --dry-run    # แสดง embed ที่จะส่ง แต่ไม่ส่ง (ใช้ทดสอบ/ถ่ายรูปใส่เล่ม)

ตั้งค่า Webhook URL (เลือกอย่างใดอย่างหนึ่ง — แนะนำ environment variable เพราะไม่ต้อง commit URL ลง git):
    $env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
    หรือใส่ใน config/plan.yaml หัวข้อ discord.webhook_url
สร้าง Webhook: Server Settings > Integrations > Webhooks > New Webhook > Copy Webhook URL

ต้องรัน `python -m main fetch-data` และ `python -m main train` มาก่อนอย่างน้อย 1 ครั้ง
"""
from __future__ import annotations

import json
import os
import sys

import requests

from src.live_signal import get_latest_signal, load_live_frame
from src.model import EnsembleModel
from src.paths import MODEL_PATH
from src.utils import get_logger, load_config

logger = get_logger("discord_alert")

COLOR_MAP = {"BUY": 0x2ECC71, "SELL": 0xE74C3C, "HOLD": 0xF1C40F}
EMOJI_MAP = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}


def get_webhook_url(config: dict) -> str | None:
    """หา Webhook URL จาก environment variable ก่อน ถ้าไม่มีค่อยดูใน config"""
    env_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if env_url:
        return env_url
    return config.get("discord", {}).get("webhook_url") or None


def build_embed(signal: dict) -> dict:
    """สร้าง Discord embed จาก dict ผลลัพธ์ของ get_latest_signal()"""
    label = signal["signal_label"]
    as_of = signal["as_of_date"].strftime("%Y-%m-%d")

    description = f"ข้อมูล ณ แท่งรายวัน {as_of} (UTC) • โมเดลทำนายทิศทางล่วงหน้า {signal['horizon']} วัน"
    if signal["risk_level"] == "HIGH":
        description += "\n⚠️ ความผันผวนสูง ควรเฝ้าระวังเป็นพิเศษ"
    if signal.get("is_stale"):
        description += f"\n⚠️ ข้อมูลค้าง {signal['data_age_days']} วัน — ตรวจสอบการดึงข้อมูล"

    return {
        "title": f"{EMOJI_MAP[label]} สัญญาณเทรด BTC/USDT: {label} ({signal['signal_text_th']})",
        "description": description,
        "color": COLOR_MAP[label],
        "fields": [
            {"name": "ราคาปิดล่าสุด", "value": f"${signal['current_price']:,.2f}", "inline": True},
            {"name": f"ความน่าจะเป็นขาขึ้น ({signal['horizon']} วัน)", "value": f"{signal['confidence_pct']:.1f}%", "inline": True},
            {"name": "Threshold", "value": f"{signal['threshold']:.2f}", "inline": True},
            {"name": "ความผันผวน (เปลี่ยนแปลง 7 วัน)", "value": f"{signal['risk_emoji']} {signal['risk_level']} ({signal['volatility_7d_pct']:+.2f}%)", "inline": True},
        ],
        "footer": {"text": "BTC/USDT ML Trading Pipeline • ทำนายทิศทางเท่านั้น ไม่ใช่คำแนะนำการลงทุน"},
    }


def send_discord_alert(webhook_url: str, embed: dict) -> bool:
    """ส่ง embed ไปที่ Discord webhook คืน True ถ้าสำเร็จ"""
    try:
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=15)
    except requests.RequestException as e:
        logger.error(f"ส่ง Discord alert ไม่สำเร็จ (เชื่อมต่อไม่ได้): {e}")
        return False
    if resp.status_code in (200, 204):
        return True
    logger.error(f"ส่ง Discord alert ไม่สำเร็จ: status={resp.status_code} body={resp.text}")
    return False


def main():
    dry_run = "--dry-run" in sys.argv
    config = load_config()

    if not MODEL_PATH.exists():
        print("ยังไม่มีโมเดลที่เทรนไว้ — รัน `python -m main train` ก่อน")
        sys.exit(1)

    try:
        live_df = load_live_frame()
        signal = get_latest_signal(live_df, EnsembleModel.load(MODEL_PATH), config)
    except (FileNotFoundError, ValueError) as e:
        print(f"คำนวณสัญญาณไม่ได้: {e}")
        sys.exit(1)

    embed = build_embed(signal)
    if dry_run:
        print(json.dumps(embed, ensure_ascii=False, indent=2))
        return

    webhook_url = get_webhook_url(config)
    if not webhook_url:
        print(
            "ไม่พบ Discord Webhook URL — ตั้งค่าก่อนใช้งานด้วยวิธีใดวิธีหนึ่ง:\n"
            "  1) $env:DISCORD_WEBHOOK_URL = \"https://discord.com/api/webhooks/...\"\n"
            "  2) หรือใส่ใน config/plan.yaml หัวข้อ discord.webhook_url\n"
            "(ทดสอบโดยไม่ส่ง: python discord_alert.py --dry-run)"
        )
        sys.exit(1)

    if send_discord_alert(webhook_url, embed):
        print(f"ส่งแจ้งเตือนสำเร็จ — {signal['signal_label']} | Prob Up: {signal['confidence_pct']:.1f}% | "
              f"ราคา: ${signal['current_price']:,.2f} | ข้อมูล ณ {signal['as_of_date'].date()}")
    else:
        print("ส่งแจ้งเตือนไม่สำเร็จ ดู log ด้านบนสำหรับรายละเอียด")
        sys.exit(1)


if __name__ == "__main__":
    main()
