"""
Bước 9: Đánh giá phân bố cosine similarity của CB Filter.
Chạy riêng: python scripts/09_eval_cb_similarity.py
Yêu cầu: scripts/02_cb_filter.py đã chạy (có models/cb_filter/product_vectors.npz)
Output: results/cb_similarity_distribution.png, results/cb_similarity_stats.json

Mục đích: Phân tích dãy giá trị cosine similarity giữa các cặp sản phẩm
để hiểu phân bố của chúng — không đánh giá model, chỉ thống kê.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, tránh lỗi GUI
import matplotlib.pyplot as plt

from src.config import MODEL_DIR, RESULT_DIR
from src.features.vectorizer import cb_similarity

# ============================================================
# Cấu hình
# ============================================================
N_SAMPLES = 200_000          # Số cặp ngẫu nhiên cần lấy mẫu
RANDOM_SEED = 42             # Seed để tái lập kết quả
CHUNK_SIZE = 20_000          # Số cặp tính 1 lần (tránh OOM)
N_BINS = 100                 # Số bins cho histogram
THRESHOLD_CB = 0.3           # Ngưỡng CB đang dùng (vẽ đường tham chiếu)

# ============================================================
# Đường dẫn
# ============================================================
VECTORS_PATH = os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")
MAPPING_PATH = os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")
OUTPUT_PNG = os.path.join(RESULT_DIR, "cb_similarity_distribution.png")
OUTPUT_JSON = os.path.join(RESULT_DIR, "cb_similarity_stats.json")

print("=" * 60)
print("  BƯỚC 9: ĐÁNH GIÁ PHÂN BỐ COSINE SIMILARITY CB FILTER")
print("=" * 60)

# ============================================================
# 1. Load dữ liệu
# ============================================================
print("\n1. Loading product vectors...")
if not os.path.exists(VECTORS_PATH):
    print(f"ERROR: Không tìm thấy {VECTORS_PATH}")
    print("Chạy scripts/02_cb_filter.py trước.")
    sys.exit(1)

product_vectors = scipy.sparse.load_npz(VECTORS_PATH)
with open(MAPPING_PATH, 'r') as f:
    product_id_to_idx = json.load(f)

n_products = product_vectors.shape[0]
print(f"   -> {n_products} products")
print(f"   -> Vector shape: {product_vectors.shape}")

# ============================================================
# 2. Lấy mẫu ngẫu nhiên N cặp
# ============================================================
print(f"\n2. Sampling {N_SAMPLES:,} random pairs...")
rng = np.random.default_rng(RANDOM_SEED)

# Sinh N_SAMPLES cặp (idx_a, idx_b) với idx_a != idx_b
idx_a = rng.integers(0, n_products, size=N_SAMPLES)
idx_b = rng.integers(0, n_products, size=N_SAMPLES)

# Đảm bảo idx_a != idx_b — nếu trùng thì random lại
mask_same = idx_a == idx_b
n_same = mask_same.sum()
while n_same > 0:
    idx_b[mask_same] = rng.integers(0, n_products, size=n_same)
    mask_same = idx_a == idx_b
    n_same = mask_same.sum()

print(f"   -> Đã sinh {N_SAMPLES:,} cặp không trùng nhau")

# ============================================================
# 3. Tính cosine similarity theo chunk
# ============================================================
print(f"\n3. Computing cosine similarities ({CHUNK_SIZE:,} pairs/chunk)...")
all_similarities = []

n_chunks = (N_SAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE
for chunk_idx in range(n_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, N_SAMPLES)

    chunk_a = idx_a[start:end]
    chunk_b = idx_b[start:end]

    # Tính similarity cho từng cặp trong chunk
    chunk_sims = []
    for i in range(len(chunk_a)):
        sim = cb_similarity(product_vectors, chunk_a[i], [chunk_b[i]])[0]
        chunk_sims.append(sim)

    all_similarities.extend(chunk_sims)

    if (chunk_idx + 1) % 5 == 0 or chunk_idx == n_chunks - 1:
        print(f"   -> Chunk {chunk_idx + 1}/{n_chunks} done ({end:,}/{N_SAMPLES:,})")

similarities = np.array(all_similarities)
print(f"   -> Hoàn thành! {len(similarities):,} similarities computed.")

# ============================================================
# 4. Thống kê
# ============================================================
print(f"\n4. Statistics:")
stats = {
    "n_samples": N_SAMPLES,
    "min": float(np.min(similarities)),
    "max": float(np.max(similarities)),
    "mean": float(np.mean(similarities)),
    "median": float(np.median(similarities)),
    "std": float(np.std(similarities)),
    "p25": float(np.percentile(similarities, 25)),
    "p50": float(np.percentile(similarities, 50)),
    "p75": float(np.percentile(similarities, 75)),
    "p90": float(np.percentile(similarities, 90)),
    "p95": float(np.percentile(similarities, 95)),
    "p99": float(np.percentile(similarities, 99)),
    "prop_above_threshold": float(np.mean(similarities >= THRESHOLD_CB)),
}

print(f"   {'Metric':<25} {'Value':<15}")
print(f"   {'-'*40}")
print(f"   {'n_samples':<25} {stats['n_samples']:<15,}")
print(f"   {'min':<25} {stats['min']:<15.6f}")
print(f"   {'max':<25} {stats['max']:<15.6f}")
print(f"   {'mean':<25} {stats['mean']:<15.6f}")
print(f"   {'median (p50)':<25} {stats['median']:<15.6f}")
print(f"   {'std':<25} {stats['std']:<15.6f}")
print(f"   {'p25':<25} {stats['p25']:<15.6f}")
print(f"   {'p75':<25} {stats['p75']:<15.6f}")
print(f"   {'p90':<25} {stats['p90']:<15.6f}")
print(f"   {'p95':<25} {stats['p95']:<15.6f}")
print(f"   {'p99':<25} {stats['p99']:<15.6f}")
print(f"   {'prop >= threshold (0.3)':<25} {stats['prop_above_threshold']:<15.4%}")

# ============================================================
# 5. Vẽ biểu đồ
# ============================================================
print(f"\n5. Plotting histogram...")
os.makedirs(RESULT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))

# Histogram
ax.hist(similarities, bins=N_BINS, range=(0, 1), alpha=0.7,
        color='steelblue', edgecolor='white', linewidth=0.5)

# Đường percentile
percentiles_to_plot = [25, 50, 75, 90, 95, 99]
percentile_colors = {
    25: ('orange', 'dashed', 'p25'),
    50: ('red', 'solid', 'p50 (median)'),
    75: ('orange', 'dashed', 'p75'),
    90: ('purple', 'dashed', 'p90'),
    95: ('purple', 'dashed', 'p95'),
    99: ('darkred', 'dashed', 'p99'),
}

for p in percentiles_to_plot:
    val = np.percentile(similarities, p)
    color, linestyle, label = percentile_colors[p]
    ax.axvline(val, color=color, linestyle=linestyle, linewidth=1.5,
               label=f'{label} = {val:.4f}')

# Đường threshold CB
ax.axvline(THRESHOLD_CB, color='green', linestyle='-', linewidth=2,
           label=f'CB threshold ({THRESHOLD_CB})')

# Đường mean
ax.axvline(stats['mean'], color='blue', linestyle=':', linewidth=1.5,
           label=f'mean = {stats["mean"]:.4f}')

ax.set_xlabel('Cosine Similarity', fontsize=12)
ax.set_ylabel('Số lượng cặp (frequency)', fontsize=12)
ax.set_title(f'Phân bố Cosine Similarity giữa các cặp sản phẩm (N={N_SAMPLES:,})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
plt.close()
print(f"   -> Saved to {OUTPUT_PNG}")

# ============================================================
# 6. Lưu stats JSON
# ============================================================
os.makedirs(RESULT_DIR, exist_ok=True)
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"   -> Saved to {OUTPUT_JSON}")

print(f"\n{'='*60}")
print(f"  DONE! Xem kết quả tại:")
print(f"  - Biểu đồ: {OUTPUT_PNG}")
print(f"  - Thống kê: {OUTPUT_JSON}")
print(f"{'='*60}")