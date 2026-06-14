"""
Buoc 9: CB Evaluation - Khao sat va danh gia Content-Based Diversity Filter.
Chay rieng: python scripts/09_cb_evaluation.py
Yeu cau: scripts 01-02 da chay (can product_vectors)

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
# CAU HINH
# ============================================================
N_PER_BIN = 10                   # so mau moi bin cho manual inspection
N_LLM_SAMPLES = 200              # so mau cho LLM survey

SAVE_DIR = os.path.join(RESULT_DIR, "cb_evaluation")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# LOAD DU LIEU & MODELS
# ============================================================
print("="*60)
print("  BUOC 9: CB EVALUATION")
print("="*60)

# Kiem tra files can thiet
checks = [
    ("CB Filter", os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")),
    ("CB Filter mapping", os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")),
    ("Products",  os.path.join(PROCESSED_DIR, "products.parquet")),
]
for name, path in checks:
    if not os.path.exists(path):
        print(f"ERROR: Thieu {name}! Chay scripts 01-02 truoc.")
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
# KHAO SAT 1: FULL SIMILARITY MATRIX - TOAN BO CAP SAN PHAM
# ============================================================
print(f"\n{'='*60}")
print(f"  KHAO SAT 1: FULL SIMILARITY MATRIX (tat ca cap san pham)")
print(f"  So san pham: {n_prod}")
print(f"{'='*60}")
t0 = time.time()

# Tinh full similarity matrix = product_vectors @ product_vectors.T
# Dung batch processing + online statistics de tranh OOM
# (1.2 ty cap = 9GB neu luu toan bo)
print("\n  Tinh full similarity matrix (online stats)...")

n_prod = cb.product_vectors.shape[0]
batch_size = 500

# Online statistics
cnt = 0
total = 0.0
total_sq = 0.0
min_val = 1.0
max_val = 0.0

# Threshold counters
threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
above_count = {t: 0 for t in threshold_list}

# Histogram bins: [0, 0.001), [0.001, 0.002), ..., [0.999, 1.0]
N_BINS = 1000
hist_counts = np.zeros(N_BINS, dtype=np.int64)

for start in range(0, n_prod, batch_size):
    end = min(start + batch_size, n_prod)
    batch_vecs = cb.product_vectors[start:end]               # (bs, D) sparse

    # similarity cua batch voi toan bo product: (bs, N) sparse
    block = batch_vecs @ cb.product_vectors.T

    for i in range(end - start):
        row_idx = start + i
        # Upper triangle: chi lay cot > row_idx
        vals = block[i, row_idx + 1:].toarray().flatten()
        if len(vals) == 0:
            continue

        # Clip ve [0, 1]
        np.clip(vals, 0, 1, out=vals)

        # Update online stats
        n = len(vals)
        cnt += n
        total += float(vals.sum())
        total_sq += float((vals * vals).sum())
        min_val = min(min_val, float(vals.min()))
        max_val = max(max_val, float(vals.max()))

        # Threshold counters
        for t in threshold_list:
            above_count[t] += int((vals >= t).sum())

        # Histogram bins (integer bin index = int(val * N_BINS))
        bin_idxs = (vals * N_BINS).astype(np.int64)
        # Clip edge case vals == 1.0 -> bin N_BINS-1
        bin_idxs = np.clip(bin_idxs, 0, N_BINS - 1)
        np.add.at(hist_counts, bin_idxs, 1)

    if (start // batch_size + 1) % 10 == 0:
        print(f"    Batch {start//batch_size + 1}/{n_prod//batch_size + 1}..."
              f" ({start}/{n_prod})")

t1 = time.time()
print(f"   Thoi gian tinh: {t1 - t0:.1f}s")
print(f"  Tong so cap (upper triangle): {cnt:,}")

# Thong ke
stats = {}
if cnt > 0:
    stats['n_pairs'] = cnt
    mean = total / cnt
    # std = sqrt(E[X^2] - E[X]^2)
    var = total_sq / cnt - mean * mean
    std = np.sqrt(max(var, 0))

    stats['mean'] = float(mean)
    stats['median'] = float(np.median(np.repeat(np.linspace(0, 1, N_BINS, endpoint=False)[:N_BINS], hist_counts)))
    stats['std'] = float(std)
    stats['min'] = float(min_val)
    stats['max'] = float(max_val)

    # Percentiles tu cumulative histogram
    cum_counts = np.cumsum(hist_counts)
    total_cnt = cum_counts[-1]
    percentiles = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        target = p / 100.0 * total_cnt
        idx = int(np.searchsorted(cum_counts, target))
        # Ensure idx is within bounds
        idx = min(idx, N_BINS - 1)
        val = (idx + 0.5) / N_BINS  # midpoint of bin
        percentiles[f'p{p}'] = float(val)
    stats['percentiles'] = percentiles

    stats['frac_above_threshold'] = {}
    for t in threshold_list:
        frac = above_count[t] / cnt
        stats['frac_above_threshold'][f'\u2265{t}'] = float(frac)

    # In thong ke
    print(f"\n  Thong ke similarity (tren toan bo {cnt:,} cap):")
    print(f"    Mean   = {stats['mean']:.6f}")
    print(f"    Median = {stats['median']:.6f}")
    print(f"    Std    = {stats['std']:.6f}")
    print(f"    Min    = {stats['min']:.6f}")
    print(f"    Max    = {stats['max']:.6f}")
    print(f"  Percentiles:")
    for k, v in stats['percentiles'].items():
        print(f"    {k:>4s} = {v:.6f}")
    print(f"  Ty le >= threshold:")
    for k, v in stats['frac_above_threshold'].items():
        print(f"    {k:>6s} = {v*100:.6f}%")

    # Ve histogram tu bin edges
    bin_edges = np.linspace(0, 1, N_BINS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram linear (full range)
    axes[0].bar(bin_edges[:-1], hist_counts, width=1/N_BINS, color='steelblue', alpha=0.8)
    axes[0].axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f"THRESHOLD={CB_THRESHOLD}")
    axes[0].set_xlabel("Cosine Similarity", fontsize=12)
    axes[0].set_ylabel("So luong cap", fontsize=12)
    axes[0].set_title(f"Phan bo Cosine Similarity (full matrix)\n{cnt:,} pairs", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    textstr = (
        f"Mean={stats['mean']:.4f}  Median={stats['median']:.4f}\n"
        f" >=0.3={stats['frac_above_threshold']['\u22650.3']*100:.2f}%  "
        f" >=0.5={stats['frac_above_threshold']['\u22650.5']*100:.2f}%  "
        f" >=0.8={stats['frac_above_threshold']['\u22650.8']*100:.2f}%"
    )
    axes[0].text(0.95, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Zoom: similarity < 0.1
    zoom_bins = int(0.1 * N_BINS)  # so bins tu 0 den 0.1
    if zoom_bins > 0:
        zoom_counts = hist_counts[:zoom_bins]
        zoom_edges = bin_edges[:zoom_bins + 1]
        axes[1].bar(zoom_edges[:-1], zoom_counts, width=1/N_BINS, color='coral', alpha=0.8)
        axes[1].axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel("Cosine Similarity", fontsize=12)
        axes[1].set_ylabel("So luong cap", fontsize=12)
        axes[1].set_title(f"Zoom: similarity < 0.1 ({zoom_counts.sum():,} pairs)", fontsize=13)
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "similarity_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Da luu: {path}")

    # Histogram log scale
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    logbins = np.logspace(np.log10(max(stats['min'], 1e-10)), np.log10(stats['max']), 80)
    ax2.hist(np.repeat(bin_edges[:-1], hist_counts), bins=logbins, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f"THRESHOLD={CB_THRESHOLD}")
    ax2.set_xscale('log')
    ax2.set_xlabel("Cosine Similarity (log scale)", fontsize=12)
    ax2.set_ylabel("So luong cap", fontsize=12)
    ax2.set_title(f"Phan bo Cosine Similarity - log scale\n{cnt:,} pairs", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(SAVE_DIR, "similarity_distribution_log.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Da luu: {path2}")
else:
    print("  WARNING: Khong co cap similarity nao!")


# ============================================================
# KHAO SAT 2: MANUAL INSPECTION
# ============================================================
t0 = time.time()
df_manual = manual_inspection_samples(
    cbfilter=cb,
    products_df=products,
    n_per_bin=N_PER_BIN,
    seed=123,          # seed khac voi survey de da dang mau
    save_dir=SAVE_DIR,
)
t1 = time.time()
print(f"   Thoi gian: {t1 - t0:.1f}s")


# ============================================================
# KHAO SAT 3: LLM SURVEY SAMPLES
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
print(f"   Thoi gian: {t1 - t0:.1f}s")


# ============================================================
# TONG HOP BAO CAO
# ============================================================
print(f"\n{'='*60}")
print(f"  TONG HOP BAO CAO")
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

# Luu report
report_path = os.path.join(SAVE_DIR, "report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n  Da luu report: {report_path}")

print(f"\n  Tat ca output duoc luu trong: {SAVE_DIR}")
print(f"  Cac file:")
for f in sorted(os.listdir(SAVE_DIR)):
    fpath = os.path.join(SAVE_DIR, f)
    size = os.path.getsize(fpath)
    print(f"    - {f} ({size/1024:.1f} KB)")

print(f"\n{'='*60}")
print(f"  HOAN THANH CB EVALUATION!")
print(f"{'='*60}")