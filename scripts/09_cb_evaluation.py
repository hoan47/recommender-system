"""
Bước 9: CB Evaluation — Khảo sát và đánh giá Content-Based Diversity Filter.
Chạy riêng: python scripts/09_cb_evaluation.py
Yêu cầu: scripts 01-07 đã chạy (cần product_vectors + ensemble model)

Output: results/cb_evaluation/
  - similarity_distribution.png
  - candidate_similarity_distribution.png
  - threshold_sweep.png + threshold_sweep.csv
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
from src.models.ochiai import OchiaiModel
from src.models.item2vec import Item2VecModel
from src.models.deepwalk import DeepWalkModel
from src.models.ensemble import EnsembleModel
from src.evaluation.cb_evaluator import (
    survey_similarity_distribution,
    survey_candidate_similarity,
    threshold_sweep,
    manual_inspection_samples,
    export_llm_survey,
)

# ============================================================
# CẤU HÌNH
# ============================================================
N_SAMPLE_PRODUCTS = 200          # số product đầu vào cho survey candidate + threshold sweep
N_SIMILARITY_SAMPLES = 10000     # số cặp ngẫu nhiên cho phân bố tổng thể
N_PER_BIN = 10                   # số mẫu mỗi bin cho manual inspection
N_LLM_SAMPLES = 200              # số mẫu cho LLM survey
TOP_K = 100                      # số candidate lấy từ ensemble

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
    ("Ochiai",    os.path.join(MODEL_DIR, "ochiai", "cooc_matrix.npz")),
    ("Item2Vec",  os.path.join(MODEL_DIR, "item2vec", "word2vec.model")),
    ("DeepWalk",  os.path.join(MODEL_DIR, "deepwalk", "embeddings.npy")),
    ("Products",  os.path.join(PROCESSED_DIR, "products.parquet")),
]
for name, path in checks:
    if not os.path.exists(path):
        print(f"ERROR: Thiếu {name}! Chạy scripts 01-07 trước.")
        print(f"  Path: {path}")
        sys.exit(1)

print("\n1. Loading data...")
products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
print(f"   {len(products)} products loaded.")

print("\n2. Loading models...")

print("   CB Filter...")
cb = CBFilter()
cb.product_vectors = scipy.sparse.load_npz(
    os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")
)
with open(os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")) as f:
    cb.product_id_to_idx = {int(k): v for k, v in json.load(f).items()}
print(f"   -> {len(cb.product_id_to_idx)} products")

print("   Ochiai...")
ochiai = OchiaiModel()
ochiai.load(os.path.join(MODEL_DIR, "ochiai"))
print(f"   -> {ochiai.n_products} products")

print("   Item2Vec...")
i2v = Item2VecModel()
i2v.load(os.path.join(MODEL_DIR, "item2vec"))
print(f"   -> {len(i2v.product_id_to_idx)} products")

print("   DeepWalk...")
deepwalk = DeepWalkModel()
deepwalk.load(os.path.join(MODEL_DIR, "deepwalk"))
print(f"   -> {len(deepwalk.product_id_to_idx)} products")

print("\n3. Initializing Ensemble...")
ensemble = EnsembleModel()
ensemble.fit(ochiai, i2v, deepwalk, cb)

# Chọn sản phẩm đầu vào cho survey (lấy mẫu từ các sản phẩm có trong tất cả models)
print("\n4. Chọn sản phẩm mẫu cho survey...")
valid_products = list(cb.product_id_to_idx.keys())
rng = np.random.RandomState(42)
product_ids = rng.choice(valid_products, size=min(N_SAMPLE_PRODUCTS, len(valid_products)),
                         replace=False).tolist()
print(f"   Chọn {len(product_ids)} sản phẩm từ {len(valid_products)} sản phẩm hợp lệ.")


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
# KHẢO SÁT 2: CANDIDATE SIMILARITY
# ============================================================
t0 = time.time()
stats2 = survey_candidate_similarity(
    cbfilter=cb,
    ensemble_model=ensemble,
    product_ids=product_ids,
    top_k=TOP_K,
    save_dir=SAVE_DIR,
)
t1 = time.time()
print(f"   ⏱ Thời gian: {t1 - t0:.1f}s")


# ============================================================
# KHẢO SÁT 3: THRESHOLD SWEEP
# ============================================================
t0 = time.time()
df_sweep = threshold_sweep(
    cbfilter=cb,
    ensemble_model=ensemble,
    product_ids=product_ids,
    top_k=TOP_K,
    save_dir=SAVE_DIR,
)
t1 = time.time()
print(f"   ⏱ Thời gian: {t1 - t0:.1f}s")

# Tìm threshold phù hợp
if not df_sweep.empty:
    # Gợi ý threshold: tại đó có 20-30% candidate bị loại
    for pct_target in [10, 20, 30, 40, 50]:
        idx = (df_sweep['avg_filtered_pct'] - pct_target).abs().idxmin()
        th = df_sweep.loc[idx, 'threshold']
        actual_pct = df_sweep.loc[idx, 'avg_filtered_pct']
        print(f"   Gợi ý: threshold≈{th:.2f} → loại ~{actual_pct:.1f}% candidate")


# ============================================================
# KHẢO SÁT 4: MANUAL INSPECTION
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
# KHẢO SÁT 5: LLM SURVEY SAMPLES
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
        'n_survey_products': len(product_ids),
        'n_similarity_samples': N_SIMILARITY_SAMPLES,
        'top_k': TOP_K,
        'ngram_range': list(cb.ngram_range),
        'max_features': cb.max_features,
    },
    'similarity_distribution': stats1,
    'candidate_similarity': stats2,
    'threshold_sweep': {
        'recommended_thresholds': {}
    },
}

# Gợi ý threshold từ sweep
if not df_sweep.empty:
    for pct_target in [10, 20, 30]:
        idx = (df_sweep['avg_filtered_pct'] - pct_target).abs().idxmin()
        th = float(df_sweep.loc[idx, 'threshold'])
        actual_pct = float(df_sweep.loc[idx, 'avg_filtered_pct'])
        report['threshold_sweep']['recommended_thresholds'][f'~{pct_target}%_filtered'] = {
            'threshold': th,
            'actual_filtered_pct': actual_pct,
        }

# Lưu report
report_path = os.path.join(SAVE_DIR, "report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n  Đã lưu report: {report_path}")

print(f"\n  Tất cả output được lưu trong: {SAVE_DIR}")
print(f"  Các file:")
for f in os.listdir(SAVE_DIR):
    fpath = os.path.join(SAVE_DIR, f)
    size = os.path.getsize(fpath)
    print(f"    - {f} ({size/1024:.1f} KB)")

print(f"\n{'='*60}")
print(f"  HOÀN THÀNH CB EVALUATION!")
print(f"{'='*60}")