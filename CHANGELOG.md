# CHANGELOG — การตรวจสอบและแก้ไขความถูกต้องของระบบ (ฉบับสหกิจ รอบสุดท้าย)

แต่ละข้อระบุ **ปัญหา → ผลกระทบ → วิธีแก้** และมีเทสต์ใน `tests/test_pipeline.py` ป้องกันไม่ให้กลับมา
(หัวข้อ "ต้องรัน train ใหม่" หมายถึงตัวเลขผลลัพธ์เก่าใช้ไม่ได้อีก ต้องรันใหม่ก่อนใส่เล่ม)

## A. บั๊กที่ทำให้สัญญาณ/ข้อมูลผิด
| # | ปัญหา | ผลกระทบ | แก้ที่ |
|---|---|---|---|
| A1 | `fetch-data` ใช้ cache CSV เสมอถ้ามีไฟล์แล้ว ไม่เช็คอายุ (และ `run_daily_alert.ps1` ไม่ได้ส่ง `--force-refresh`) | แจ้งเตือนรายวันใช้ข้อมูลของวันที่รันครั้งแรกตลอดไป | `data_loader.py`: ตรวจอายุ cache ต่อ symbol (BTC/ETH ≤1 วัน, DXY/VIX ≤4 วัน, on-chain ≤3 วัน) ดึงใหม่อัตโนมัติ ถ้าดึงไม่ได้ใช้ cache เดิมพร้อมเตือน (ยกเว้น BTC ใน `--force-refresh` จะล้มเหลวชัดเจน); `run_daily_alert.ps1` ใช้ `--force-refresh` + เขียน log + ไม่ส่ง alert ถ้าดึง BTC สดไม่ได้ |
| A2 | สัญญาณ "ล่าสุด" ใช้แถวสุดท้ายของ `feature_matrix.csv` ซึ่งถูก `dropna()` ตัดท้าย `target_horizon` (7) แถวทิ้ง | ราคา, ความผันผวน, proba_up ที่ Discord/dashboard แสดงเป็นของ **7 วันก่อน** | `features.build_live_matrix()` สร้าง `data/live_features.csv` ที่ยาวถึงแท่งล่าสุดโดยไม่ต้องมี target; `live_signal.py`/`app.py`/`discord_alert.py` อ่านไฟล์นี้ |
| A3 | แท่งรายวันของ "วันนี้" ที่ยังไม่ปิด (Binance/yfinance คืนมาด้วย) ถูกใช้เป็นข้อมูลจริง | ฟีเจอร์/สัญญาณคำนวณจากราคากลางวัน และเปลี่ยนค่าไปมาตามเวลาที่รัน | `drop_incomplete_last_bar()` ตัดแท่ง UTC วันนี้ออกทุกชุดข้อมูล |
| A4 | `min_confidence_gap: 0.05` ซ้ำซ้อนกับ `threshold: 0.55` (เทียบเท่ากันพอดี) แต่คอมเมนต์ใน yaml เขียนว่า "ค่าต่ำสุดที่ gap มีผลจริง" | อธิบายพารามิเตอร์ผิดในเล่มได้ | แก้คอมเมนต์ + docstring; `live_signal` ใช้ `signals_from_confidence()` ตัวเดียวกับ backtest เพื่อไม่ให้ตรรกะสองที่ไม่ตรงกัน |
| A5 | ไฟล์ `oos_proba.csv` สั้นกว่า feature matrix หลัง `fetch-data` รอบใหม่ → `df.loc[boolean_series]` โยน `IndexingError` ใน dashboard/backtest/sweep | พังหลังรัน daily alert | `src/evaluation.py` (`load_oos`/`align_oos`) จับคู่ด้วย "วันที่" ใช้ในทุกสคริปต์ |
| A6 | `merge_cross_assets` ใช้ `ffill().bfill()` | bfill เติมช่วงต้นข้อมูลด้วยค่าจาก "อนาคต" (look-ahead เล็กน้อย) | เปลี่ยนเป็น `ffill().dropna()` |
| A7 | Binance: ใช้ timezone เครื่อง (UTC+7) คำนวณเวลาเริ่ม/จบ; `RSI` คืน NaN เมื่อไม่มีวันลงเลยในหน้าต่าง; label ท้ายข้อมูลเป็น 0 ปลอม (ก่อน dropna) | ความคลาดเคลื่อนเล็กน้อย / แถวถูกตัดโดยไม่จำเป็น | UTC ทั้งหมด, RSI=100 ในกรณีนั้น, target เป็น NaN ถ้ายังไม่รู้อนาคต |

## B. ระเบียบวิธี (กระทบตัวเลขผลลัพธ์ — ต้องรัน train ใหม่)
| # | ปัญหา | แก้ที่ |
|---|---|---|
| B1 | Walk-forward ไม่มี **purge/embargo**: label ของแถวท้าย train มองราคาไปข้างหน้า 7 วัน ซึ่งล้ำเข้าไปในช่วง test (และช่วง val ที่ใช้ tune) → OOS ดีเกินจริงเล็กน้อย | `walk_forward_splits(..., purge=)` + `model.walk_forward.purge_days: 7` ใช้ที่ train↔test, train↔val, และโมเดลสุดท้าย |
| B2 | ฟีเจอร์เป็น **ระดับราคาดิบ** (`bb_upper/mid/lower`, `atr`, `macd*`) — non-stationary (BTC $20k→$100k+) ต้นไม้เรียนรู้ "ช่วงเวลา" แทนรูปแบบ | ตัดออกจาก feature set (เก็บไว้ให้ rule-based ใช้) เพิ่มเวอร์ชัน normalize: `bb_width`, `close_to_bb_mid`, `macd_pct`, `macd_signal_pct`, `macd_hist_pct` |
| B3 | โมเดลไม่จำรายชื่อฟีเจอร์ → แก้ `features.py` แล้วลืมเทรนใหม่ จะ predict ผิดเงียบๆ หรือ error ไม่ชัด | `EnsembleModel.feature_cols` บันทึกตอน fit, `get_latest_signal` ตรวจและแจ้ง "รัน train ใหม่" |

## C. Backtester / การประเมินผล
- ปิดสถานะที่ค้างท้ายข้อมูล (`end_of_data`) พร้อมค่าธรรมเนียม — เดิมเทรดสุดท้ายไม่ถูกนับในสถิติ
- เพิ่ม metric: Sortino, Calmar, Profit Factor, Annual return, Exposure %, Avg holding days, `pnl_net_pct`
- เพิ่ม **Buy & Hold** เป็นเส้นฐานผ่าน engine เดียวกัน (ในตาราง `compare` และ dashboard)
- `generate_report.py`: ตาราง/กราฟ/สรุปไทย + block-bootstrap CI ของ Sharpe + ความเสถียรรายช่วงเวลา + ความไวต่อพารามิเตอร์ + calibration + p-value (ปรับ label ทับซ้อน)

## D. อื่นๆ
- `discord_alert.py --dry-run`, ข้อความแจ้งเตือนไม่บอก "ตรวจพบความผันผวน" ทุกครั้งอีกต่อไป, เตือนเมื่อข้อมูลค้าง
- `app.py`: การ์ดราคา/RSI/regime ใช้ข้อมูลล่าสุดจริง, เตือนข้อมูลค้าง, ไม่พังทั้งหน้าถ้าโมเดล/ฟีเจอร์ไม่ตรงกัน
- `feature_importance.py`: ใช้รายชื่อฟีเจอร์จากตัวโมเดล, ป้ายกราฟเป็นอังกฤษ (ฟอนต์ไทยไม่ขึ้นใน matplotlib บน Windows)
- เพิ่ม `requirements-dev.txt` (pytest), `matplotlib==3.9.2` ใน requirements, `.gitignore`
