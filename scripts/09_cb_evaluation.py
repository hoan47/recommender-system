"""
Bước 9: CB Evaluation — Khảo sát và đánh giá Content-Based Diversity Filter.
Chạy riêng: python scripts/09_cb_evaluation.py
Yêu cầu: scripts 01-02 đã chạy (cần product_vectors)

Output: results/cb_evaluation/
  - similarity_distribution.png
  - manual_samples.csv
  - llm_survey_samples.csv
  - report.json
"""
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse

from src.config import MODEL_DIR, PROCESSED_DIR, RESULT_DIR
from src.models.cb_filter import CBFilter
from src.evaluation.cb_evaluator import (
    survey_similarity_distribution,
    manual_inspection_samples,
    export_llm_survey,
)

# ============================================================
# CẤU HÌNH
# ============================================================
N_SIMILARITY_SAMPLES = 10000     # số cặp ngẫu nhiên cho phân bố tổng thể
N_PER_BIN = 10                   # số mẫu mỗi bin cho manual inspection
N_LLM_SAMPLES = 200              # số mẫu cho LLM survey

SAVE_DIR = os.path.join(RESULT_DIR, "cb_evaluation")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# LOAD DỮ LIỆU & MODELS
# ============================================================
print("="*60)
print("  BƯỚC 9: CB EVALUATION")
print("="*60)

# Kiểm tra files cần thiết
checks = [
    ("CB Filter", os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")),
    ("CB Filter mapping", os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")),
    ("Products",  os.path.join(PROCESSED_DIR, "products.parquet")),
]
for name, path in checks:
    if not os.path.exists(path):
        print(f"ERROR: Thiếu {name}! Chạy scripts 01-02 trước.")
        print(f"  Path: {path}")
        sys.exit(1)

print("\n1. Loading data...")
products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
print(f"   {len(products)} products loaded.")

print("\n2. Loading CB Filter...")
cb = CBFilter()
cb.product_vectors = scipy.sparse.load_npz(
    os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")
)
with open(os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")) as f:
    cb.product_id_to_idx = {int(k): v for k, v in json.load(f).items()}
print(f"   -> {len(cb.product_id_to_idx)} products")


# ============================================================
# KHẢO SÁT 1: PHÂN BỐ SIMILARITY TỔNG THỂ
# ============================================================
print("\n" + "="*60)
t0 = time.time()
stats1 = survey_similarity_distribution(
    cbfilter=cb,
    products_df=products,
    n_samples=N_SIMILARITY_SAMPLES,
    seed=42,
    save_dir=SAVE_DIR,
)
t1 = time.time()
print(f"   ⏱ Thời gian: {t1 - t0:.1f}s")


# ============================================================
# KHẢO SÁT 2: MANUAL INSPECTION
# ============================================================
t0 = time.time()
df_manual = manual_inspection_samples(
    cbfilter=cb,
    products_df=products,
    n_per_bin=N_PER_BIN,
    seed=123,          # seed khác với survey để đa dạng mẫu
    save_dir=SAVE_DIR,
)
t1 = time.time()
print(f"   ⏱ Thời gian: {t1 - t0:.1f}s")


# ============================================================
# KHẢO SÁT 3: LLM SURVEY SAMPLES
# ============================================================
t0 = time.time()
df_llm = export_llm_survey(
    cbfilter=cb,
    products_df=products,
    n_samples=N_LLM_SAMPLES,
    sim_min=0.2,
    sim_max=0.5,
    seed=42,
    save_dir=SAVE_DIR,
)
t1 = time.time()
print(f"   ⏱ Thời gian: {t1 - t0:.1f}s")


# ============================================================
# TỔNG HỢP BÁO CÁO
# ============================================================
print(f"\n{'='*60}")
print(f"  TỔNG HỢP BÁO CÁO")
print(f"{'='*60}")

report = {
    'config': {
        'current_threshold': cb.threshold,
        'n_products': len(cb.product_id_to_idx),
        'n_similarity_samples': N_SIMILARITY_SAMPLES,
        'ngram_range': list(cb.ngram_range),
        'max_features': cb.max_features,
    },
    'similarity_distribution': stats1,
}

# Lưu report
report_path = os.path.join(SAVE_DIR, "report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n  Đã lưu report: {report_path}")

print(f"\n  Tất cả output được lưu trong: {SAVE_DIR}")
print(f"  Các file:")
for f in sorted(os.listdir(SAVE_DIR)):
    fpath = os.path.join(SAVE_DIR, f)
    size = os.path.getsize(fpath)
    print(f"    - {f} ({size/1024:.1f} KB)")

print(f"\n{'='*60}")
print(f"  HOÀN THÀNH CB EVALUATION!")
print(f"{'='*60}")