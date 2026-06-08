"""
Hybrid — kết hợp Confidence + Knowledge Graph, loại substitute bằng CB

Công thức:
  final_score(A → B) = α * Conf_norm(A,B) + β * KG_sim(A,B)
  Nếu CB_sim(A,B) > threshold → final_score = 0 (loại substitute)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import math
import numpy as np
from scipy.sparse import load_npz, save_npz, csr_matrix, lil_matrix, coo_matrix
from tqdm import tqdm

from src.config import MODELS_DIR
from src.config import HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH

HYBRID_FILE = MODELS_DIR / "hybrid_matrix.npz"
CONF_FILE = MODELS_DIR / "confidence_matrix.npz"


def build_cb_sparse(cb_vectors, n_products):
    """Xây dựng sparse feature matrix từ cb_vectors dict-based."""
    if not cb_vectors:
        return csr_matrix((n_products, 0), dtype=np.float32)

    rows, cols, vals = [], [], []
    for pid, vec in cb_vectors.items():
        if pid >= n_products:
            continue
        for vocab_idx, val in vec.items():
            rows.append(pid)
            cols.append(vocab_idx)
            vals.append(val)

    if not rows:
        return csr_matrix((n_products, 0), dtype=np.float32)

    n_vocab = max(cols) + 1
    feat = csr_matrix((vals, (rows, cols)), shape=(n_products, n_vocab), dtype=np.float32)
    del rows, cols, vals; gc.collect()
    return feat


def build_cb_similarity(cb_feat, top_k=100, chunk=2000):
    """CB similarity matrix = feat @ feat.T (cosine similarity vì đã L2-norm)."""
    n = cb_feat.shape[0]
    if n == 0 or cb_feat.nnz == 0:
        return csr_matrix((n, n), dtype=np.float32)

    print("  [Hybrid] Computing CB similarity matrix ...")
    sim = lil_matrix((n, n), dtype=np.float32)

    for start in tqdm(range(0, n, chunk), desc="  CB similarity"):
        end = min(start + chunk, n)
        chunk_sim = cb_feat[start:end].dot(cb_feat.T)

        for i_local, row_idx in enumerate(range(start, end)):
            row_dense = chunk_sim[i_local].toarray().ravel()
            row_dense[row_idx] = 0
            if row_dense.size == 0:
                continue

            if top_k < n and row_dense.size > top_k:
                idx = np.argpartition(row_dense, -top_k)[-top_k:]
                vals = row_dense[idx]
            else:
                idx = np.arange(row_dense.size, dtype=np.int32)
                vals = row_dense

            pos = vals > 0
            pos_idx = idx[pos] if idx.dtype == np.int32 else idx[pos]
            if np.any(pos):
                sim[row_idx, pos_idx] = vals[pos].astype(np.float32)

    csr = sim.tocsr()
    del sim; gc.collect()
    print(f"  [Hybrid] CB similarity: {csr.nnz:,} entries")
    return csr


def build_hybrid(confidence, kg_sim, cb_vectors,
                 alpha=HYBRID_ALPHA, beta=HYBRID_BETA, cb_thresh=HYBRID_CB_THRESH):
    """
    Kết hợp Confidence và KG thành hybrid score, loại substitute bằng CB filter.

    Quy trình:
        1. Normalize Confidence về [0, 1]
        2. Compute combined scores: α * Conf_norm + β * KG_sim (sparse addition)
        3. Precompute CB similarity matrix sparse từ cb_vectors
        4. Với mỗi dòng, zero-out entries có CB similarity > threshold
        5. Giữ top scoring entries
    """
    print(f"\n  [Hybrid] α={alpha}, β={beta}, cb_thresh={cb_thresh} ...")
    n = confidence.shape[0]

    # Bước 1: Normalize Confidence về [0, 1]
    conf_data = confidence.data.copy()
    max_s = conf_data.max()
    if max_s > 0:
        conf_data /= max_s
    conf_norm = csr_matrix((conf_data, confidence.indices, confidence.indptr), shape=confidence.shape)

    # Bước 2: Combine Confidence + KG
    print("  [Hybrid] Combining Confidence + KG scores ...")
    combined = alpha * conf_norm + beta * kg_sim
    del conf_norm; gc.collect()
    print(f"  [Hybrid] Combined: {combined.nnz:,} entries")

    if combined.nnz == 0:
        print("  [Hybrid] Warning: combined matrix empty!")
        return combined

    cb_sim = None
    if cb_vectors and cb_thresh < 1.0:
        cb_feat = build_cb_sparse(cb_vectors, n)
        if cb_feat.nnz > 0:
            cb_sim = build_cb_similarity(cb_feat, top_k=200)
            del cb_feat; gc.collect()

    if cb_sim is not None and cb_sim.nnz > 0:
        print(f"  [Hybrid] Applying CB filter (threshold={cb_thresh}) ...")

        hybrid = combined.tolil()
        cb_coo = cb_sim.tocoo()

        row_set = {}
        for i, j, v in tqdm(zip(cb_coo.row, cb_coo.col, cb_coo.data),
                            desc="  CB filter", total=cb_coo.nnz):
            if v > cb_thresh and i < n and j < n:
                if i not in row_set:
                    row_set[i] = set()
                row_set[i].add(j)

        for i in tqdm(row_set, desc="  Removing substitutes"):
            for j in row_set[i]:
                hybrid[i, j] = 0.0

        del cb_sim, row_set; gc.collect()
        csr_result = hybrid.tocsr()
        del hybrid; gc.collect()
    else:
        csr_result = combined

    del combined; gc.collect()
    print(f"  [Hybrid] Done: {csr_result.nnz:,} entries")
    return csr_result


def save(matrix):
    save_npz(HYBRID_FILE, matrix)
    print(f"  [Hybrid] Saved: {HYBRID_FILE}")


def load():
    return load_npz(HYBRID_FILE)


if __name__ == "__main__":
    from src.features.build_cb import load as load_cb

    print("  [Hybrid] Loading Confidence matrix ...")
    confidence = load_npz(CONF_FILE)

    print("  [Hybrid] Loading KG similarity ...")
    kg_sim = load_npz(MODELS_DIR / "kg_similarity.npz")

    print("  [Hybrid] Loading CB vectors ...")
    load_cb()
    from src.features.build_cb import cb_vectors

    h = build_hybrid(confidence, kg_sim, cb_vectors)
    save(h)