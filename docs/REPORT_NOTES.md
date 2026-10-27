# คู่มือประกอบการเขียนเล่มสหกิจ

## 0) ต้องทำก่อนเขียนตัวเลขลงเล่ม
โค้ดถูกแก้ระเบียบวิธี (purge, ฟีเจอร์ stationary, ตัดแท่งที่ยังไม่ปิด) → **ตัวเลขผลลัพธ์จากเวอร์ชันก่อนหน้าใช้ไม่ได้ ต้องรันใหม่**

```powershell
.\setup.ps1                                # ถ้ายังไม่มี venv / ต้องติดตั้ง matplotlib เพิ่ม: pip install -r requirements.txt
python -m main fetch-data --force-refresh  # ข้อมูลล่าสุด + live_features.csv
.\train_with_backup.ps1                    # เทรนใหม่ (สำรองของเก่าไว้ให้)
python -m main backtest
python -m main compare
python -m main report                      # ได้ results\report\ ทั้งหมด
python sweep_threshold.py; python sweep_stop_loss.py; python sweep_per_strategy_tp.py   # (ถ้าอยากจูนใหม่ — ดูข้อ 4)
python -m pytest -q                        # ยืนยันว่าระบบผ่านเทสต์ (pip install -r requirements-dev.txt)
```

## 1) โครงเล่มที่แนะนำ และไฟล์ที่ใช้ประกอบ
| บท | เนื้อหา | ใช้จาก |
|---|---|---|
| 1 บทนำ | ที่มา วัตถุประสงค์ ขอบเขต (ทำนายทิศทาง 7 วัน, long-only, รายวัน) | README ส่วนภาพรวม |
| 2 ทฤษฎี/เครื่องมือ | indicators (RSI, MACD, BBands, ATR), ensemble soft-voting, Walk-Forward + purge, Optuna, Sharpe/Sortino/Max DD | `features.py`, `model.py` |
| 3 การออกแบบระบบ | สถาปัตยกรรม (แผนภาพใน README), data pipeline, feature set, backtest engine, dashboard, Discord | README, `CHANGELOG.md` หมวด A–B |
| 4 ผลการทดลอง | ตารางเทียบกลยุทธ์, equity/drawdown, accuracy/AUC/calibration, ความเสถียรรายช่วงเวลา, ความไวต่อพารามิเตอร์ | `results/report/*` |
| 5 สรุป/ข้อจำกัด/งานต่อยอด | ตามจริง ดูข้อ 3 และ 5 ด้านล่าง | `REPORT_SUMMARY.md` ข้อ 4–5 |
| ภาคผนวก | การตรวจสอบความถูกต้อง: บั๊กที่พบ/แก้ + เทสต์ | `CHANGELOG.md`, `tests/` |

กราฟสำคัญ → ตำแหน่งในเล่ม: `equity_curves.png` (4), `drawdown.png` (4), `walk_forward_accuracy.png` (3–4 อธิบาย overfitting),
`calibration.png` (4), `sensitivity_sharpe.png` (4), `feature_importance.png` (4), `exit_reasons.png` (4), `price_with_trades.png` (4)
ตารางหลัก: `strategy_comparison.md` / `segment_stability.csv` / `walk_forward_results.csv`
(ป้ายในกราฟเป็นภาษาอังกฤษ ใส่คำอธิบายภาษาไทยในแคปชันของเล่มได้เลย)

## 2) ไทม์ไลน์การพัฒนา (เรียบเรียงจากงานที่ผ่านมา — ใช้เป็นเนื้อหาบทที่ 3 ส่วนกระบวนการพัฒนา)
- เริ่มจากแนวคิด ML ensemble บน PAXG/USDT → ย้ายไป S&P 500 (ก.ค. 2569) → ปรับเป็น BTC/USDT พร้อม rule-based baselines เป็นตัวเทียบวินิจฉัย
- ส.ค.: แก้ปัญหาสภาพแวดล้อม Windows (DLL ถูกบล็อก, execution policy, ล็อกเวอร์ชัน Click/pandas); ขยาย `target_horizon` เป็น 7 วัน (ผลเปลี่ยนจากขาดทุนเป็นกำไรในเวอร์ชันนั้น); fix seed ของ Optuna และ `start_date` ให้ทดลองซ้ำได้; แก้ overfitting หนักด้วย L1/L2 + บทลงโทษ train–val gap
- ก.ย.: แก้ Sharpe annualization เป็น √365 (crypto เทรดทุกวัน); พบบั๊ก stop-loss ตรวจจากราคาปิดอย่างเดียวทำให้ขาดทุนจริงลึกกว่าที่ตั้ง (ตั้ง 5% แต่เจอ −18%) → เปลี่ยนเป็นตรวจจาก low/high และออกที่ราคา SL/TP เป๊ะ
- ต.ค.: sweep TP × sizing, TP รายกลยุทธ์, stop-loss → ค่าใน `plan.yaml` ปัจจุบัน (threshold 0.55, SL 3%, TP 8%, dynamic sizing)
- 24–26 ต.ค.: audit ครั้งสุดท้าย → `CHANGELOG.md` (A1–A7, B1–B3, C)

> ตัวเลขผลลัพธ์ในแต่ละช่วงข้างต้นมาจากโค้ดเวอร์ชันก่อน audit — ห้ามยกมาเป็นผลสุดท้ายโดยไม่รันใหม่
> แต่ "เรื่องเล่า" ของการค้นพบและแก้บั๊กเป็นเนื้อหาที่ดีมากสำหรับเล่ม เพราะแสดงกระบวนการตรวจสอบเชิงวิทยาศาสตร์

## 3) เขียนผลให้ซื่อสัตย์ — ตัวอย่างประโยค (เลือกให้ตรงกับผลที่รันได้จริง)
- ถ้า ML Sharpe < baseline ง่ายๆ: *"เมื่อเทียบแบบแฟร์ (แต่ละกลยุทธ์ใช้ TP ที่เหมาะสมของตัวเอง) MACD Crossover ให้ Sharpe สูงกว่า ML Ensemble จึงสรุปว่าความซับซ้อนของโมเดลยังไม่สร้างความได้เปรียบที่ชัดเจนกับข้อมูลรายวันชุดนี้ ผลงานหลักของโครงการคือระบบประเมินผลที่ปราศจาก data leakage ซึ่งทำให้ข้อสรุปนี้เชื่อถือได้"*
- ถ้า CI ของ Sharpe ครอบคลุม 0: *"ช่วงความเชื่อมั่น 95% ของ Sharpe จาก block bootstrap ครอบคลุมศูนย์ จึงยังไม่สามารถปฏิเสธได้ว่าผลตอบแทนเกิดจากโชค"*
- ถ้า accuracy ใกล้ baseline แต่ backtest ดี: *"ความแม่นยำของทิศทางใกล้เคียงการเดา แต่การจัดการความเสี่ยง (SL/TP, dynamic sizing, ถือสถานะ X% ของเวลา) ลด Max Drawdown ได้ X% เมื่อเทียบ Buy & Hold"*
- เรื่อง selection bias: *"พารามิเตอร์ถูกเลือกจากผล OOS เดียวกับที่รายงาน ตัวเลขจึงเป็นขอบบนของประสิทธิภาพที่คาดหวัง"* — เขียนตรงๆ ดีกว่าให้กรรมการจับได้เอง

## 4) ถ้าจะ sweep พารามิเตอร์ใหม่ — ข้อควรระวัง
พารามิเตอร์ที่เลือกจากผล OOS แล้วรายงานผล OOS เดิม จะดีเกินจริงเสมอ (ยิ่งไล่หลายค่ายิ่งแย่) ทางเลือกที่ซื่อสัตย์:
(ก) ใช้ค่าใน `plan.yaml` เดิมโดยไม่จูนใหม่ แล้วรายงานตามที่ได้ + ตาราง `sensitivity_sharpe` แสดงว่าผลไม่พึ่งค่าเดียว
(ข) จูนบนช่วงต้นของ OOS แล้วรายงานเฉพาะช่วงท้าย (เช่น fold 5) — ต้องบอกในเล่มว่าแบ่งอย่างไร
(ค) forward test: เก็บสัญญาณ Discord รายวันจากนี้ไปแล้วเทียบกับราคาจริง (ข้อมูลนี้ไม่เคยถูกใช้จูน)

## 5) คำถามที่กรรมการมักถาม — แนวตอบ
| คำถาม | แนวตอบ / หลักฐาน |
|---|---|
| ทำไมไม่ทำนายราคาแต่ทำนายทิศทาง? | ราคาจริงเป็น non-stationary, regression ของราคาให้ตัวเลขที่ดูดีหลอกตา (R² สูงเพราะ persistence); ทิศทางประเมินด้วย accuracy/AUC เทียบ baseline ได้ตรงไปตรงมา |
| ป้องกัน data leakage อย่างไร? | Walk-Forward (expanding) + purge 7 วัน + backtest เฉพาะ OOS + ฟีเจอร์คำนวณย้อนหลังเท่านั้น + เทสต์ `test_walk_forward_*`, `test_training_target_*` |
| ผลนี้เชื่อได้แค่ไหน? | ดู bootstrap CI, ความเสถียรรายช่วงเวลา, p-value แบบปรับ label ทับซ้อน และข้อจำกัด selection bias (ข้อ 3–4) |
| ทำไม ML ไม่ชนะ/ชนะ rule-based? | ใช้ตาราง `strategy_comparison` + `feature_importance` อธิบาย; ถ้าไม่ชนะ ให้พูดตรงๆ ว่าเป็นผลการทดลองที่ได้ |
| ค่าธรรมเนียม/slippage? | 0.1% + 0.05% ต่อด้าน ใน `plan.yaml`; ลองเพิ่มเป็นสองเท่าเพื่อทดสอบความทนทานได้ |
| นำไปใช้จริงได้ไหม? | ยังเป็นระบบแจ้งเตือน (ไม่ส่งคำสั่งซื้อขายเอง); ขั้นต่อไปคือ paper trading + monitoring drift |

## 6) งานต่อยอด (ใช้เขียนบทสรุป)
paper trading ผ่าน exchange testnet · ฟีเจอร์เพิ่ม (funding rate, open interest, sentiment) · ตรวจจับ concept drift แล้ว retrain อัตโนมัติ ·
calibrate ความน่าจะเป็น (isotonic/Platt) ก่อนใช้กับ dynamic sizing · ทดสอบหลายสินทรัพย์ · วัดผลด้วย purged k-fold / CPCV
