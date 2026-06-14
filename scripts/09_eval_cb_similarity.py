"""
Bước 9: Phân tích phân bố cosine similarity CB Filter.
Output: results/cb_similarity_stats.json, results/cb_similarity_distribution.png
"""
import json, os, sys, pandas as pd, numpy as np, scipy.sparse, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_DIR, RESULT_DIR, PROCESSED_DIR
from src.features.vectorizer import cb_similarity

# === Load vectors ===
v = scipy.sparse.load_npz(os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz"))
with open(os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")) as f:
    pid_to_idx = json.load(f)
n = v.shape[0]

# === Load products để có tên ===
p = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
idx_to_pid = {v: int(k) for k, v in pid_to_idx.items()}
pid_to_name = dict(zip(p['product_id'], p['product_name']))
idx_to_name = {i: pid_to_name.get(pid_to_idx[int(k)], "?") for i, k in enumerate(pid_to_idx)}

# === Thống kê: 200K cặp ngẫu nhiên ===
rng = np.random.default_rng(42)
a = rng.integers(0, n, 200_000)
b = rng.integers(0, n, 200_000)
m = a == b
while m.sum():
    b[m] = rng.integers(0, n, m.sum())
    m = a == b
sim = np.array([cb_similarity(v, a[i], [b[i]])[0] for i in range(200_000)])

stats = {"n_samples": 200_000, "min": float(sim.min()), "max": float(sim.max()),
         "mean": float(sim.mean()), "median": float(np.median(sim)), "std": float(sim.std()),
         "p25": float(np.percentile(sim, 25)), "p50": float(np.percentile(sim, 50)),
         "p75": float(np.percentile(sim, 75)), "p90": float(np.percentile(sim, 90)),
         "p95": float(np.percentile(sim, 95)), "p99": float(np.percentile(sim, 99))}
for k, v in stats.items():
    print(f"  {k:20s} = {v:.6f}" if k != "n_samples" else f"  {k:20s} = {v:,}")

# === Đối chiếu thực tế ===
def pair(ka, kb):
    """Tìm product chứa keyword, tính similarity"""
    try:
        ia = int(p[p['product_name'].str.contains(ka, case=False)]['product_id'].iloc[0])
        ib = int(p[p['product_name'].str.contains(kb, case=False)]['product_id'].iloc[0])
        if ia == ib and len(p[p['product_name'].str.contains(kb, case=False)]) > 1:
            ib = int(p[p['product_name'].str.contains(kb, case=False)]['product_id'].iloc[1])
        s = cb_similarity(v, pid_to_idx[str(ia)], [pid_to_idx[str(ib)]])[0]
        print(f"  {pid_to_name[ia]:<50s} | {pid_to_name[ib]:<50s} | {s:.4f}")
    except: pass

print("\n--- SUBSTITUTE (cùng loại) ---")
for kw in ['beer', 'milk', 'bread', 'cheese', 'chicken']:
    pair(kw, kw)
print("\n--- COMPLEMENTARY (mua kèm) ---")
for ka, kb in [('beer','chip'), ('milk','cereal'), ('bread','peanut butter'), ('pasta','sauce'), ('coffee','cream')]:
    pair(ka, kb)
print("\n--- KHÔNG LIÊN QUAN ---")
for ka, kb in [('beer','diaper'), ('milk','shampoo'), ('bread','soap'), ('chicken','toothpaste'), ('banana','tire')]:
    pair(ka, kb)

# === Lưu kết quả ===
os.makedirs(RESULT_DIR, exist_ok=True)
with open(os.path.join(RESULT_DIR, "cb_similarity_stats.json"), 'w') as f:
    json.dump(stats, f, indent=2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sim, bins=100, range=(0, 1), color='steelblue', edgecolor='white', alpha=0.7)
for p, c, ls in [(50, 'red', '-'), (75, 'orange', '--'), (90, 'purple', ':'), (99, 'darkred', ':')]:
    ax.axvline(np.percentile(sim, p), color=c, linestyle=ls, label=f'p{p}={np.percentile(sim, p):.3f}')
ax.axvline(0.3, color='green', ls='-', lw=2, label='threshold (0.3)')
ax.set_xlabel('Cosine similarity'); ax.set_ylabel('Frequency'); ax.legend()
plt.savefig(os.path.join(RESULT_DIR, "cb_similarity_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nDone! Kết quả tại {RESULT_DIR}/")