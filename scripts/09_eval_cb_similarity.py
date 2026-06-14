"""
Bước 9: Phân tích phân bố cosine similarity CB Filter (full matrix, CPU).
RAM chỉ lưu histogram 10 bins + ma trận sparse.
Dùng sparse matrix multiplication → chỉ duyệt nonzero elements.
"""
import json, os, sys
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_DIR, RESULT_DIR, CB_THRESHOLD

# === Load vectors ===
print("Đang load product_vectors...")
v = sparse.load_npz(os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")).tocsr()
n = v.shape[0]
print(f"  shape: {v.shape}")

# === Chuẩn hóa L2 từng hàng (unit norm) --> cosine = dot ===
print("Đang chuẩn hóa L2...")
norm = np.sqrt((v.multiply(v)).sum(axis=1)).A1
norm[norm == 0] = 1e-9
v = v.multiply(1.0 / norm[:, np.newaxis]).tocsr()

# === Tính sparse similarity matrix (chỉ nonzero) ===
print("Đang tính v @ v.T (sparse)...")
sim = v.dot(v.T).tocoo()
total_nz = len(sim.data)
print(f"  nonzero elements: {total_nz:,}")

# === Duyệt nonzero, lấy upper triangle, đếm histogram ===
bins = [round(i * 0.1, 1) for i in range(10)]
counts = [0] * 10
nz_upper = 0

print("Đang duyệt upper triangle...")
for i, j, val in zip(sim.row, sim.col, sim.data):
    if j > i:  # upper triangle
        val = float(np.clip(val, 0, 1))
        bin_idx = min(int(val * 10), 9)
        counts[bin_idx] += 1
        nz_upper += 1

total_pairs = n * (n - 1) // 2
print(f"  upper triangle pairs: {total_pairs:,}")
print(f"  nonzero upper pairs:  {nz_upper:,}")
print(f"  pairs bị bỏ qua (cos=0): {total_pairs - nz_upper:,}")

# === Kết quả ===
dist = {f"{b:.1f}-{b+0.1:.1f}": c for b, c in zip(bins, counts)}
result = {
    "bins": [f"{b:.1f}-{b+0.1:.1f}" for b in bins],
    "counts": counts,
    "total_pairs": total_pairs,
    "nonzero_pairs": nz_upper,
    "threshold": CB_THRESHOLD,
    "distribution": dist
}
print("\nPhân bố cosine similarity (step 0.1):")
for k, v in dist.items():
    pct = v * 100 / nz_upper if nz_upper else 0
    print(f"  [{k:7s})  {v:10,} cặp ({pct:.2f}%)")

# === Lưu JSON ===
os.makedirs(RESULT_DIR, exist_ok=True)
json_path = os.path.join(RESULT_DIR, "cb_similarity_histogram.json")
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nĐã lưu JSON: {json_path}")

# === Vẽ biểu đồ ===
fig, ax = plt.subplots(figsize=(10, 6))
bar_labels = [f"{b:.1f}" for b in bins]
bar_centers = np.arange(len(bins)) + 0.05
ax.bar(bar_centers, counts, width=0.09, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0.3, color='green', ls='-', lw=2, label=f'threshold={CB_THRESHOLD}')
ax.set_xticks([b + 0.05 for b in range(10)])
ax.set_xticklabels(bar_labels)
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Số cặp (nonzero)')
ax.set_title('Phân bố cosine similarity (full matrix, step 0.1)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

png_path = os.path.join(RESULT_DIR, "cb_similarity_histogram.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Đã lưu biểu đồ: {png_path}")
print("Hoàn tất!")