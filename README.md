# BTC/USDT ML Trading Pipeline

Pipeline ทำนาย **ทิศทาง** ราคา BTC/USDT ล่วงหน้า 7 วันด้วย ensemble (XGBoost + LightGBM + RandomForest, soft-voting)
ประเมินด้วย Walk-Forward Validation + out-of-sample backtest เทียบกับ rule-based strategies และ Buy & Hold
พร้อม dashboard (Streamlit), แจ้งเตือนรายวันผ่าน Discord และสคริปต์สร้างตาราง/กราฟสำหรับใส่รายงาน

> ระบบนี้ทำนายทิศทางขึ้น/ลง ไม่ใช่ราคาที่แน่นอน และเป็นงานวิจัย/การศึกษา ไม่ใช่คำแนะนำการลงทุน

## ภาพรวมการทำงาน

```
Binance API ─┐
yfinance (ETH, DXY, VIX) ─┼─► data_loader ─► features ─┬─► feature_matrix.csv  (มี target → เทรน/backtest)
Blockchain.com (on-chain) ─┘   (cache + เช็คอายุ)       └─► live_features.csv   (ถึงแท่งล่าสุด → สัญญาณล่าสุด)

feature_matrix ─► model (Walk-Forward + purge + Optuna) ─┬─► oos_proba.csv      (ทำนายนอกตัวอย่างเท่านั้น)
                                                          └─► ensemble_model.joblib (ใช้กับข้อมูลใหม่)

oos_proba ─► backtester ─► compare (ML vs rule-based vs Buy&Hold) ─► generate_report ─► results/report/
live_features + model ─► live_signal ─┬─► app.py (dashboard)
                                      └─► discord_alert.py (แจ้งเตือนรายวัน)
```

หลักการออกแบบที่สำคัญ (ทุกข้อมีเทสต์รองรับ):
1. **Backtest ใช้เฉพาะ out-of-sample** — ไม่เอาโมเดลสุดท้ายมา predict ข้อมูลที่มันเทรนมา
2. **Purge/embargo** ระหว่าง train / val / test เท่ากับ `target_horizon` กัน label leakage
3. **ฟีเจอร์ stationary** — ไม่ใช้ระดับราคาดิบ (BBands/ATR/MACD แบบ absolute) เป็น feature
4. **สัญญาณล่าสุดใช้ข้อมูลล่าสุดจริง** (`live_features.csv`) และใช้ฟังก์ชันตัดสินใจเดียวกับ backtest
5. **เทียบแฟร์** — ทุกกลยุทธ์รันผ่าน engine เดียวกัน ช่วงเวลาเดียวกัน มีค่าธรรมเนียม/slippage และ Buy & Hold เป็นเส้นฐาน

## โครงสร้างไฟล์

```
config/plan.yaml          ค่าตั้งทั้งหมด (threshold, SL/TP, sizing, walk-forward, Optuna, ...)
src/data_loader.py        ดึงข้อมูล + cache + เช็คอายุ + ตัดแท่งที่ยังไม่ปิด
src/features.py           indicators, regime, cross-asset, on-chain, target, build_feature_matrix / build_live_matrix
src/rule_strategies.py    baseline: RSI Reversal, MACD Crossover, Volatility Breakout
src/model.py              EnsembleModel, Walk-Forward (purge), Optuna
src/backtester.py         engine (pure pandas), metrics, Buy & Hold, compare_strategies
src/live_signal.py        สัญญาณล่าสุด BUY/HOLD/SELL (ใช้ร่วม dashboard + Discord)
src/evaluation.py         จับคู่ oos_proba กับ feature matrix ด้วยวันที่
src/paths.py, utils.py    path กลาง, config, logging
main.py                   CLI (Typer): fetch-data / train / backtest / compare / report / all
generate_report.py        ตาราง + กราฟ + สรุปไทย สำหรับใส่เล่ม
app.py                    Streamlit dashboard
discord_alert.py          แจ้งเตือน Discord (--dry-run ทดสอบได้)
feature_importance.py     เทียบ feature importance ข้ามโมเดล
analyze_exits.py          วิเคราะห์เหตุผลที่ออกจากสถานะ (SL / TP / signal)
sweep_*.py                ไล่พารามิเตอร์: threshold, stop-loss, TP×sizing, TP ต่อกลยุทธ์
test_pipeline_synthetic.py  ทดสอบทั้ง pipeline ด้วยข้อมูลจำลอง
tests/                    pytest (ไม่ต้องใช้อินเทอร์เน็ต)
*.ps1                     setup / run_pipeline / run_dashboard / run_daily_alert / train_with_backup / restore_backup
```

## ติดตั้งและรัน (Windows PowerShell)

```powershell
.\setup.ps1                 # ครั้งเดียว: สร้าง venv + ติดตั้ง dependency (ล็อกเวอร์ชันไว้กัน DLL ถูกบล็อก)
.\run_pipeline.ps1          # fetch -> train -> backtest -> compare -> report  (train ใช้เวลานานสุด)
.\run_dashboard.ps1         # เปิด dashboard
```

ถ้ารัน `.ps1` ไม่ได้: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`

### คำสั่งทีละขั้น (activate venv ก่อน: `.\venv\Scripts\Activate.ps1`)
| คำสั่ง | ทำอะไร |
|---|---|
| `python -m main fetch-data [--force-refresh]` | ดึงข้อมูล (ใช้ cache ถ้ายังสด) สร้าง `feature_matrix.csv` + `live_features.csv` |
| `python -m main train` | เทรน Walk-Forward + Optuna → `models/`, `data/oos_proba.csv`, `results/walk_forward_results.csv` |
| `.\train_with_backup.ps1` | เหมือน train แต่สำรองโมเดลเก่าก่อน (กู้คืนด้วย `restore_backup.ps1`) |
| `python -m main backtest` | backtest ML บน OOS → `results/ml_backtest_summary.csv`, `ml_trade_log.csv`, `ml_equity_curve.csv` |
| `python -m main compare` | ML vs rule-based vs Buy & Hold → `results/strategy_comparison.csv` |
| `python -m main report` | สร้างทุกอย่างสำหรับเล่มที่ `results/report/` (เริ่มอ่าน `REPORT_SUMMARY.md`) |
| `python -m streamlit run app.py` | dashboard (ห้าม `python app.py` ตรงๆ) |
| `python discord_alert.py [--dry-run]` | ส่ง/ทดสอบการแจ้งเตือน |
| `python -m pytest -q` | เทสต์ (ติดตั้งก่อน: `pip install -r requirements-dev.txt`) |

## แจ้งเตือนรายวันผ่าน Discord
1. ตั้ง Webhook: `$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."` (หรือใส่ใน `plan.yaml` — แนะนำ env var เพราะไม่ควร commit URL)
2. ทดสอบ: `python discord_alert.py --dry-run` แล้วค่อยรันจริง
3. Task Scheduler: Program `powershell.exe`, Arguments `-ExecutionPolicy Bypass -File "...\run_daily_alert.ps1"` เวลา **07:30 น.** ขึ้นไป (แท่งรายวัน Binance ปิด 00:00 UTC = 07:00 เวลาไทย)
   สคริปต์ดึงข้อมูลสดทุกครั้ง, ไม่ส่ง alert ถ้าดึงข้อมูล BTC สดไม่ได้ (สินทรัพย์อื่นใช้ cache สำรอง) และบันทึก `logs\daily_alert.log`

## ค่าตั้งสำคัญใน `config/plan.yaml`
| ค่า | ความหมาย |
|---|---|
| `model.target_horizon` | ทำนายทิศทางล่วงหน้ากี่วัน (7) |
| `model.walk_forward.purge_days` | ช่องว่างกัน label leakage ต้อง ≥ target_horizon |
| `backtest.confidence_threshold` | proba_up ≥ ค่านี้ = ซื้อ, ≤ 1−ค่านี้ = ปิดสถานะ |
| `backtest.min_confidence_gap` | |proba−0.5| ขั้นต่ำ — ซ้ำซ้อนกับ threshold เมื่อ threshold ≥ 0.5+gap |
| `backtest.stop_loss_pct / take_profit_pct` | เช็คจาก low/high รายวัน ออกที่ราคา SL/TP เป๊ะ (วันเดียวโดนทั้งคู่ = SL ก่อน) |
| `backtest.position_sizing_mode` | `fixed` หรือ `dynamic` (ขนาดตามความมั่นใจระหว่าง min–max) |

## ข้อจำกัดที่ต้องรู้ (ระบุในเล่มด้วย)
- พารามิเตอร์ (threshold, SL, TP) ถูกเลือกจาก sweep บนผล OOS ชุดเดียวกับที่รายงาน → มี selection bias ต้องยืนยันด้วย forward test
- ข้อมูลรายวัน ไม่ใช่ tick-level; backtest ไม่จำลอง order book / partial fill / latency
- label 7 วันทับซ้อนกัน → จำนวนตัวอย่างอิสระจริงน้อยกว่าจำนวนแถว (รายงานคำนวณ p-value แบบปรับให้แล้ว)
- โมเดล tree-based ยัง overfit train ได้สูง (ดู Overfitting Monitor) แม้ใส่ regularization + penalty ใน Optuna แล้ว
- ตลาด crypto เปลี่ยน regime เร็ว ต้อง monitor และ retrain เป็นระยะ

## แก้ปัญหาที่พบบ่อย
| ปัญหา | วิธีแก้ |
|---|---|
| `missing ScriptRunContext` | ใช้ `python -m streamlit run app.py` |
| `n_trades: 0` | ลด `confidence_threshold` |
| `pip is not recognized` | activate venv: `.\venv\Scripts\Activate.ps1` |
| `DLL load failed ... Application Control policy` | ลบ `venv` แล้วรัน `.\setup.ps1` ใหม่ (requirements ล็อกเวอร์ชันที่ปลอดภัยไว้) |
| dashboard/discord แจ้ง "โมเดลใช้ฟีเจอร์ที่ไม่มีในข้อมูล" | แก้ `features.py` หลังเทรนล่าสุด → `python -m main train` ใหม่ |
| ข้อความ "ข้อมูลค้าง N วัน" | `python -m main fetch-data` (หรือ `--force-refresh`) |
| ดึง yfinance ไม่ได้ | เช็คอินเทอร์เน็ต; ระบบใช้ cache เดิมแทนพร้อม warning |

ประวัติการแก้ไขและบั๊กที่พบทั้งหมด: [`CHANGELOG.md`](CHANGELOG.md) · แนวทางเขียนเล่ม: [`docs/REPORT_NOTES.md`](docs/REPORT_NOTES.md)
