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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import MODEL_DIR, PROCESSED_DIR, RESULT_DIR, CB_THRESHOLD
from src.models.cb_filter import CBFilter
from src.evaluation.cb_evaluator import (
    manual_inspection_samples,
    export_llm_survey,
)

# ============================================================
# CẤU HÌNH
# ============================================================
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
n_prod = len(cb.product_id_to_idx)
print(f"   -> {n_prod} products")


# ============================================================
# KHẢO SÁT 1: FULL SIMILARITY MATRIX — TOÀN BỘ CẶP SẢN PHẨM
# ============================================================
print(f"\n{'='*60}")
print(f"  KHẢO SÁT 1: FULL SIMILARITY MATRIX (tất cả cặp sản phẩm)")
print(f"  Số sản phẩm: {n_prod}")
print(f"{'='*60}")
t0 = time.time()

# Tính full similarity matrix = product_vectors @ product_vectors.T
# product_vectors: CSR (n_prod, 15K) sparse
# Kết quả: CSR (n_prod, n_prod) — similarity giữa mọi cặp
print("\n  Tính full similarity matrix...")
sim_matrix = cb.product_vectors @ cb.product_vectors.T
print(f"  Sim matrix shape: {sim_matrix.shape}")
print(f"  Sim matrix non-zero: {sim_matrix.nnz}")

# Lấy upper triangle (k=1) để tránh duplicate + diagonal
# tril k=-1 lấy lower triangle không gồm diagonal
lower = scipy.sparse.tril(sim_matrix, k=-1, format='csr')
all_values = lower.data
print(f"  Upper triangle (unique pairs): {len(all_values):,}")

t1 = time.time()
print(f"   ⏱ Thời gian tính: {t1 - t0:.1f}s")

# Thống kê
stats = {}
if len(all_values) > 0:
    stats['n_pairs'] = int(len(all_values))
    stats['mean'] = float(np.mean(all_values))
    stats['median'] = float(np.median(all_values))
    stats['std'] = float(np.std(all_values))
    stats['min'] = float(np.min(all_values))
    stats['max'] = float(np.max(all_values))
    stats['percentiles'] = {
        'p1': float(np.percentile(all_values, 1)),
        'p5': float(np.percentile(all_values, 5)),
        'p10': float(np.percentile(all_values, 10)),
        'p25': float(np.percentile(all_values, 25)),
        'p50': float(np.percentile(all_values, 50)),
        'p75': float(np.percentile(all_values, 75)),
        'p90': float(np.percentile(all_values, 90)),
        'p95': float(np.percentile(all_values, 95)),
        'p99': float(np.percentile(all_values, 99)),
    }
    stats['frac_above_threshold'] = {}
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        frac = float(np.mean(all_values >= t))
        stats['frac_above_threshold'][f'≥{t}'] = frac

    # In thống kê
    print(f"\n  Thống kê similarity (trên toàn bộ {len(all_values):,} cặp):")
    print(f"    Mean   = {stats['mean']:.4f}")
    print(f"    Median = {stats['median']:.4f}")
    print(f"    Std    = {stats['std']:.4f}")
    print(f"    Min    = {stats['min']:.6f}")
    print(f"    Max    = {stats['max']:.6f}")
    print(f"  Percentiles:")
    for k, v in stats['percentiles'].items():
        print(f"    {k:>4s} = {v:.6f}")
    print(f"  Tỷ lệ >= threshold:")
    for k, v in stats['frac_above_threshold'].items():
        print(f"    {k:>6s} = {v*100:.4f}%")

    # Histogram — log scale vì đa số similarity rất thấp
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram linear (full range)
    axes[0].hist(all_values, bins=100, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f"THRESHOLD={CB_THRESHOLD}")
    axes[0].set_xlabel("Cosine Similarity", fontsize=12)
    axes[0].set_ylabel("Số lượng cặp", fontsize=12)
    axes[0].set_title(f"Phân bố Cosine Similarity (full matrix)\n{len(all_values):,} pairs", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    textstr = (
        f"Mean={stats['mean']:.4f}  Median={stats['median']:.4f}\n"
        f"≥0.3={stats['frac_above_threshold']['≥0.3']*100:.2f}%  "
        f"≥0.5={stats['frac_above_threshold']['≥0.5']*100:.2f}%  "
        f"≥0.8={stats['frac_above_threshold']['≥0.8']*100:.2f}%"
    )
    axes[0].text(0.95, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Zoom vào vùng similarity thấp (0-0.1) để thấy rõ phân bố
    zoom_mask = all_values < 0.1
    if zoom_mask.sum() > 0:
        zoom_vals = all_values[zoom_mask]
        axes[1].hist(zoom_vals, bins=80, color='coral', edgecolor='white', alpha=0.8)
        axes[1].axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel("Cosine Similarity", fontsize=12)
        axes[1].set_ylabel("Số lượng cặp", fontsize=12)
        axes[1].set_title(f"Zoom: similarity < 0.1 ({len(zoom_vals):,} pairs)", fontsize=13)
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "similarity_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Đã lưu: {path}")

    # Histogram của phân bố trên thang log
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    logbins = np.logspace(np.log10(max(stats['min'], 1e-10)), np.log10(stats['max']), 80)
    ax2.hist(all_values, bins=logbins, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f"THRESHOLD={CB_THRESHOLD}")
    ax2.set_xscale('log')
    ax2.set_xlabel("Cosine Similarity (log scale)", fontsize=12)
    ax2.set_ylabel("Số lượng cặp", fontsize=12)
    ax2.set_title(f"Phân bố Cosine Similarity — log scale\n{len(all_values):,} pairs", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(SAVE_DIR, "similarity_distribution_log.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Đã lưu: {path2}")
else:
    print("  WARNING: Không có non-zero similarity nào!")


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
        'n_products': n_prod,
        'ngram_range': list(cb.ngram_range),
        'max_features': cb.max_features,
    },
    'similarity_distribution': stats,
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