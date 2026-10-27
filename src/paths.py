"""
paths.py — path กลางของไฟล์ข้อมูล/โมเดล/ผลลัพธ์ ที่ทุกสคริปต์ใช้ร่วมกัน
(เดิมแต่ละสคริปต์ประกาศ path ของตัวเองซ้ำๆ กัน ถ้าจะเปลี่ยนชื่อไฟล์ต้องไล่แก้หลายที่)
"""
from src.utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# feature matrix สำหรับเทรน/backtest: มี target ครบ จึงถูกตัดท้าย target_horizon แถวทิ้ง
FEATURE_DATA_PATH = DATA_DIR / "feature_matrix.csv"
# feature สำหรับ "สัญญาณล่าสุด": ไม่ต้องมี target จึงยาวถึงแท่งที่ปิดล่าสุดจริง
LIVE_FEATURE_PATH = DATA_DIR / "live_features.csv"
OOS_PROBA_PATH = DATA_DIR / "oos_proba.csv"
MODEL_PATH = MODELS_DIR / "ensemble_model.joblib"
WF_RESULTS_PATH = RESULTS_DIR / "walk_forward_results.csv"
REPORT_DIR = RESULTS_DIR / "report"
