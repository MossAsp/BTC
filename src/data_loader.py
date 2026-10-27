"""
data_loader.py — ดึงข้อมูลราคา BTC/USDT, cross-asset, และ on-chain metrics
BTC ดึงจาก Binance public API เป็นหลัก (เร็ว ไม่มี rate limit เข้มงวดแบบ yfinance)
ถ้า Binance ใช้ไม่ได้ (เช่น network บล็อก) จะ fallback ไปใช้ yfinance แทนอัตโนมัติ
cross-asset (ETH, DXY, VIX) ยังใช้ yfinance เหมือนเดิม เพราะ Binance ไม่มีข้อมูลตลาด traditional finance
on-chain metrics (unique addresses, hash rate) ดึงจาก Blockchain.com Charts API (ฟรี ไม่ต้องใช้ API key)
บันทึกเป็นไฟล์ CSV แยกตาม symbol ใน data/ (ตั้งชื่อไฟล์ตาม symbol เพื่อไม่ทับกัน)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src.utils import PROJECT_ROOT, get_logger, load_config

logger = get_logger("data_loader")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BLOCKCHAIN_CHARTS_URL = "https://api.blockchain.info/charts"

# ชื่อ chart ของ Blockchain.com -> ชื่อคอลัมน์ที่จะใช้ในโปรเจกต์
ONCHAIN_CHARTS = {
    "n-unique-addresses": "onchain_active_addresses",
    "hash-rate": "onchain_hash_rate",
}


def _safe_symbol_filename(symbol: str) -> str:
    """แปลง ticker เช่น '^VIX' หรือ 'DX-Y.NYB' ให้เป็นชื่อไฟล์ที่ปลอดภัย"""
    return symbol.replace("^", "").replace(".", "_").replace("-", "_")


def fetch_btc_from_binance(start_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    ดึงข้อมูล BTC/USDT จาก Binance public REST API โดยตรง (ไม่ต้องใช้ API key)
    Binance จำกัดคืนค่าสูงสุด 1000 แท่งต่อ request จึงต้องวนดึงเป็นช่วงๆ จนถึงปัจจุบัน
    """
    interval_map = {"1d": "1d", "1h": "1h", "1wk": "1w"}
    binance_interval = interval_map.get(interval, "1d")

    # ใช้ UTC ล้วน (เดิมใช้ timezone เครื่อง ทำให้เวลาเริ่มเพี้ยน 7 ชม. บนเครื่องที่เป็นเวลาไทย)
    start_ts = int(dt.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    end_ts = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)

    all_rows = []
    cursor = start_ts
    while cursor < end_ts:
        params = {
            "symbol": "BTCUSDT",
            "interval": binance_interval,
            "startTime": cursor,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        # เลื่อน cursor ไปหลัง timestamp ล่าสุดที่ได้มา 1 แท่ง กันดึงซ้ำ
        cursor = rows[-1][0] + 1
        if len(rows) < 1000:
            break

    if not all_rows:
        raise ValueError("Binance ไม่คืนข้อมูลใดๆ มาเลย")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]
    return df


def fetch_symbol(symbol: str, interval: str, start: str | None, lookback_days: int) -> pd.DataFrame:
    """ดึงข้อมูล OHLCV ของ symbol เดียวจาก yfinance"""
    if start is None:
        start_date = (dt.datetime.now() - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    else:
        start_date = start

    logger.info(f"กำลังดึงข้อมูล {symbol} ตั้งแต่ {start_date} (interval={interval})")
    df = yf.download(symbol, start=start_date, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"ไม่พบข้อมูลสำหรับ {symbol} — เช็ค ticker หรือช่วงวันที่อีกครั้ง")

    # yfinance บางเวอร์ชันคืน MultiIndex columns เวลาดึงหลาย symbol ในทีเดียว, กันไว้เผื่อ
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def fetch_onchain_metric(chart_name: str, start_date: str) -> pd.Series:
    """
    ดึง on-chain metric เดียวจาก Blockchain.com Charts API (ฟรี ไม่ต้องใช้ API key)
    คืนค่าเป็น pd.Series (index=date) เพื่อเอาไป join กับ DataFrame หลักได้ง่าย
    """
    params = {"timespan": "all", "format": "json", "sampled": "false"}
    url = f"{BLOCKCHAIN_CHARTS_URL}/{chart_name}"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    values = payload.get("values", [])
    if not values:
        raise ValueError(f"Blockchain.com ไม่คืนข้อมูลใดๆ มาสำหรับ {chart_name}")

    dates = [dt.datetime.fromtimestamp(v["x"], dt.timezone.utc).date() for v in values]
    series = pd.Series([v["y"] for v in values], index=pd.to_datetime(dates), name=chart_name)
    series = series[~series.index.duplicated(keep="last")]
    series = series.loc[series.index >= pd.to_datetime(start_date)]
    return series


def fetch_onchain_metrics(start_date: str) -> pd.DataFrame:
    """
    ดึง on-chain metric ทั้งหมดที่กำหนดไว้ใน ONCHAIN_CHARTS แล้วรวมเป็น DataFrame เดียว
    ถ้าตัวไหนดึงไม่สำเร็จ จะข้ามตัวนั้นไป (ไม่ทำให้ pipeline หยุดทำงาน เหมือน cross-asset)
    """
    columns = {}
    for chart_name, col_name in ONCHAIN_CHARTS.items():
        try:
            series = fetch_onchain_metric(chart_name, start_date)
            columns[col_name] = series
            logger.info(f"ดึง on-chain metric {chart_name} สำเร็จ ({len(series)} แถว)")
        except Exception as e:
            logger.warning(f"ดึง on-chain metric {chart_name} ไม่สำเร็จ ({e}) — ข้ามตัวนี้ไป")

    if not columns:
        return pd.DataFrame()

    df = pd.DataFrame(columns)
    df.index.name = "date"
    return df


# อายุ cache สูงสุด (วัน) ก่อนถือว่าเก่าเกินไปและดึงใหม่อัตโนมัติ
# BTC/ETH เทรดทุกวัน -> แท่งที่ปิดล่าสุดคือเมื่อวาน (UTC) จึงเก่าได้ไม่เกิน 1 วัน
# DXY/VIX หยุดเสาร์-อาทิตย์และวันหยุด -> เผื่อ 4 วัน | on-chain จาก Blockchain.com ล่าช้า 1-2 วันเป็นปกติ
MAX_CACHE_AGE_DAYS = {"btc": 1, "eth": 1, "dxy": 4, "vix": 4, "onchain": 3}
DEFAULT_MAX_CACHE_AGE_DAYS = 4


def _today_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def drop_incomplete_last_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    ตัดแท่งรายวันของ "วันนี้ (UTC)" ที่ยังไม่ปิดออก — ทั้ง Binance และ yfinance คืนแท่งปัจจุบันที่ยังวิ่งอยู่มาด้วย
    ถ้าปล่อยไว้ ฟีเจอร์/สัญญาณล่าสุดจะคำนวณจากราคากลางวันที่ยังเปลี่ยนต่อ และ backtest ก็เห็นราคาปิดปลอม
    """
    if df is None or df.empty:
        return df
    return df[df.index < _today_utc()]


def _is_stale(df: pd.DataFrame | None, key: str) -> bool:
    """cache เก่าเกินกว่าที่ยอมรับได้ไหม (เดิมไม่เช็คเลย -> fetch-data รอบสองเป็นต้นไปไม่อัปเดตอะไร)"""
    if df is None or df.empty:
        return True
    age_days = (_today_utc() - pd.Timestamp(df.index.max()).normalize()).days
    return age_days > MAX_CACHE_AGE_DAYS.get(key, DEFAULT_MAX_CACHE_AGE_DAYS)


def _download_symbol(key: str, symbol: str, data_cfg: dict) -> pd.DataFrame | None:
    """ดึงข้อมูลสดของ symbol เดียว (BTC: Binance ก่อนแล้ว fallback yfinance) คืน None ถ้าล้มเหลว"""
    start_date = data_cfg.get("start_date") or (
        dt.datetime.now() - dt.timedelta(days=data_cfg["lookback_days"])
    ).strftime("%Y-%m-%d")

    if key == "btc":
        try:
            df = fetch_btc_from_binance(start_date, data_cfg["interval"])
            logger.info(f"ดึงข้อมูล BTC จาก Binance สำเร็จ ({len(df)} แถว)")
            return df
        except Exception as e:
            logger.warning(f"ดึงข้อมูล BTC จาก Binance ไม่สำเร็จ ({e}) — ลอง yfinance แทน")

    try:
        return fetch_symbol(symbol, data_cfg["interval"], data_cfg.get("start_date"), data_cfg["lookback_days"])
    except Exception as e:
        logger.warning(f"ดึงข้อมูล {symbol} ไม่สำเร็จ ({e})")
        return None


def load_all_data(config: dict, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    ดึงข้อมูลหลัก (BTC), cross-asset และ on-chain ตาม config
    คืน dict: {"btc": df, "eth": df, "dxy": df, "vix": df, "onchain": df}

    นโยบาย cache: ใช้ไฟล์ใน data/ ถ้า "ยังสดพอ" (ดู MAX_CACHE_AGE_DAYS) ไม่งั้นดึงใหม่อัตโนมัติ
    ถ้าดึงใหม่ไม่สำเร็จ (เน็ตหลุด) จะใช้ cache เดิมแทนพร้อม warning ดีกว่าให้ pipeline ล้ม
    """
    data_cfg = config["data"]
    data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    daily = data_cfg["interval"] == "1d"

    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        return drop_incomplete_last_bar(frame) if daily else frame

    symbols = {"btc": data_cfg["symbol"], **data_cfg.get("cross_assets", {})}
    result: dict[str, pd.DataFrame] = {}

    for key, symbol in symbols.items():
        fname = data_dir / f"{_safe_symbol_filename(symbol)}.csv"
        cached = None
        df = None

        if fname.exists():
            cached = _clean(pd.read_csv(fname, index_col="date", parse_dates=True))
            if force_refresh:
                logger.info(f"force refresh: ดึง {symbol} ใหม่ (เก็บ cache ไว้เป็นตัวสำรองถ้าดึงไม่ได้)")
            elif _is_stale(cached, key):
                logger.info(f"cache ของ {symbol} เก่าเกินไป (ล่าสุด {cached.index.max().date()}) — ดึงใหม่")
            else:
                logger.info(f"โหลด {symbol} จาก cache: {fname}")
                df = cached

        if df is None:
            fresh = _download_symbol(key, symbol, data_cfg)
            if fresh is not None:
                df = _clean(fresh)
                df.to_csv(fname)
                logger.info(f"บันทึก {symbol} -> {fname} ({len(df)} แถว, ล่าสุด {df.index.max().date()})")
            elif cached is not None:
                if force_refresh and key == "btc":
                    # งานรายวัน (force refresh) ต้องการข้อมูล BTC ล่าสุดจริง ถ้าดึงไม่ได้ให้ล้มเหลวชัดเจน
                    # ดีกว่าส่งสัญญาณจากข้อมูลเก่าโดยไม่รู้ตัว (สินทรัพย์อื่นยังใช้ cache สำรองได้)
                    raise RuntimeError("force refresh: ดึงข้อมูล BTC ล่าสุดไม่สำเร็จ — หยุดเพื่อไม่ให้ใช้ข้อมูลเก่า")
                logger.warning(f"ใช้ cache เดิมของ {symbol} แทน (ล่าสุด {cached.index.max().date()})")
                df = cached
            else:
                logger.warning(f"ข้าม {symbol} — ไม่มีทั้งข้อมูลสดและ cache")
                continue
        result[key] = df

    # on-chain: หลาย metric ในไฟล์เดียว
    onchain_fname = data_dir / "onchain_metrics.csv"
    onchain_df = None
    cached_on = None
    if onchain_fname.exists():
        cached_on = _clean(pd.read_csv(onchain_fname, index_col="date", parse_dates=True))
        if force_refresh:
            logger.info("force refresh: ดึง on-chain ใหม่ (เก็บ cache ไว้สำรอง)")
        elif _is_stale(cached_on, "onchain"):
            logger.info("cache on-chain เก่าเกินไป — ดึงใหม่")
        else:
            logger.info(f"โหลด on-chain metrics จาก cache: {onchain_fname}")
            onchain_df = cached_on
    if onchain_df is None:
        start_date_for_onchain = data_cfg.get("start_date") or (
            dt.datetime.now() - dt.timedelta(days=data_cfg["lookback_days"])
        ).strftime("%Y-%m-%d")
        fresh_on = fetch_onchain_metrics(start_date_for_onchain)
        if not fresh_on.empty:
            onchain_df = _clean(fresh_on)
            onchain_df.to_csv(onchain_fname)
            logger.info(f"บันทึก on-chain metrics -> {onchain_fname} ({len(onchain_df)} แถว)")
        elif cached_on is not None:
            logger.warning("ดึง on-chain ใหม่ไม่สำเร็จ — ใช้ cache เดิมแทน")
            onchain_df = cached_on
        else:
            onchain_df = pd.DataFrame()
    if not onchain_df.empty:
        result["onchain"] = onchain_df

    if "btc" not in result:
        raise RuntimeError("ดึงข้อมูล BTC หลักไม่สำเร็จ — pipeline ทำงานต่อไม่ได้")

    return result


def merge_cross_assets(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    รวมข้อมูล BTC กับ cross-asset เข้าด้วยกันเป็น DataFrame เดียว
    ใช้ BTC เป็นแกนหลัก (BTC เทรด 24/7 แต่หุ้น/ดัชนีมหภาคเทรดแค่วันทำการ จึง forward-fill ค่าที่ขาด)
    """
    btc = data["btc"].copy()
    btc.columns = [f"btc_{c}" for c in btc.columns]

    merged = btc.copy()
    for key, df in data.items():
        if key == "btc":
            continue
        if key == "onchain":
            # on-chain metrics มีหลายคอลัมน์อยู่แล้ว (ไม่ใช่แค่ close) join เข้าไปตรงๆ
            merged = merged.join(df, how="left")
        else:
            close_col = df[["close"]].rename(columns={"close": f"{key}_close"})
            merged = merged.join(close_col, how="left")

    # forward-fill (BTC เทรด 24/7 แต่หุ้น/ดัชนีหยุดเสาร์-อาทิตย์) แล้วตัดแถวต้นไฟล์ที่ยังไม่มีค่าทิ้ง
    # (เดิมใช้ bfill ซึ่งเอาค่าจาก "อนาคต" มาเติมช่วงต้นข้อมูล — look-ahead เล็กน้อย)
    n_before = len(merged)
    merged = merged.ffill().dropna()
    if len(merged) < n_before:
        logger.info(f"ตัดแถวต้นข้อมูลที่ cross-asset ยังไม่มีค่าออก {n_before - len(merged)} แถว")
    return merged


if __name__ == "__main__":
    cfg = load_config()
    raw = load_all_data(cfg)
    df = merge_cross_assets(raw)
    print(df.tail())
    print(f"\nรวมข้อมูลทั้งหมด: {len(df)} แถว, คอลัมน์: {list(df.columns)}")
