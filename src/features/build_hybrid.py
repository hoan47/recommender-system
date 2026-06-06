"""
Hybrid — kết hợp SPMI + KG, loại substitute bằng CB
final_score = α * SPMI + β * KG, filter nếu CB sim > threshold
"""
import gc
import numpy as np
from scipy.sparse import load_npz, save_npz, csr_matrix, lil_matrix
from tqdm import tqdm

from src.config import MODELS_DIR
from src.config import HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH

HYBRID_FILE = MODELS_DIR / "hybrid_matrix.npz"

def build_hybrid(spmi, kg_sim, cb_vectors, alpha=HYBRID_ALPHA, beta=HYBRID_BETA, cb_thresh=HYBRID_CB_THRESH):
    """
    Kết hợp SPMI + KG, dùng CB filter.
    spmi, kg_sim: csr_matrix (n×n)
    cb_vectors: dict {pid: {idx: val}} từ build_cb
    """
    print(f"\n  [Hybrid] α={alpha}, β={beta}, cb_thresh={cb_thresh} ...")
    n = spmi.shape[0]

    # Normalize SPMI về [0, 1]
    spmi_data = spmi.data.copy()
    max_s = spmi_data.max()
    if max_s > 0:
        spmi_data /= max_s
    spmi_norm = csr_matrix((spmi_data, spmi.indices, spmi.indptr), shape=spmi.shape)

    hybrid = lil_matrix((n, n), dtype=np.float32)

    for i in tqdm(range(n), desc="  Hybrid"):
        # Gom SPMI + KG scores
        scores = {}
        spmi_row = spmi_norm[i]
        if spmi_row.nnz:
            for j, v in zip(spmi_row.indices, spmi_row.data):
                scores[j] = alpha * v
        kg_row = kg_sim[i]
        if kg_row.nnz:
            for j, v in zip(kg_row.indices, kg_row.data):
                scores[j] = scores.get(j, 0) + beta * v

        if not scores:
            continue

        # CB filter: loại sản phẩm quá giống
        cb_vec_i = cb_vectors.get(i, {})
        filtered = {}
        for j, s in scores.items():
            if s <= 0:
                continue
            # Kiểm tra CB similarity
            cb_vec_j = cb_vectors.get(j, {})
            if cb_vec_i and cb_vec_j:
                if len(cb_vec_i) > len(cb_vec_j):
                    va, vb = cb_vec_j, cb_vec_i
                else:
                    va, vb = cb_vec_i, cb_vec_j
                cb_sim = sum(va[t] * vb[t] for t in va if t in vb)
                if cb_sim > cb_thresh:
                    continue  # substitute, loại bỏ
            filtered[j] = s

        if filtered:
            items = list(filtered.items())
            items.sort(key=lambda x: -x[1])
            hybrid[i, [j for j, _ in items]] = [s for _, s in items]

    csr = hybrid.tocsr()
    del hybrid; gc.collect()
    print(f"  [Hybrid] Done: {csr.nnz:,} entries")
    return csr

def save(matrix):
    save_npz(HYBRID_FILE, matrix)
    print(f"  [Hybrid] Saved: {HYBRID_FILE}")

def load():
    return load_npz(HYBRID_FILE)

if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, str(MODELS_DIR.parent))
    from src.features.build_cb import load as load_cb
    spmi = load_npz(MODELS_DIR / "spmi_matrix.npz")
    kg_sim = load_npz(MODELS_DIR / "kg_similarity.npz")
    load_cb()
    from src.features.build_cb import cb_vectors
    h = build_hybrid(spmi, kg_sim, cb_vectors)
    save(h)