"""
Hybrid — kết hợp Confidence + Knowledge Graph, loại substitute bằng CB

Công thức:
  final_score(A → B) = α * Conf_norm(A,B) + β * KG_sim(A,B)
  Nếu CB_sim(A,B) > threshold → final_score = 0 (loại substitute)

Optimizations:
  1. Xây dựng CB feature matrix sparse (n_products × vocab_size) từ cb_vectors
  2. CB similarity = feature_matrix @ feature_matrix.T (sparse dot product)
  3. Batch combine Confidence + KG bằng sparse matrix operations
  4. CB filter = element-wise mask trên combined scores
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

# File lưu hybrid matrix
HYBRID_FILE = MODELS_DIR / "hybrid_matrix.npz"
CONF_FILE = MODELS_DIR / "confidence_matrix.npz"


def build_cb_sparse(cb_vectors, n_products):
    """
    Xây dựng sparse feature matrix từ cb_vectors.
    
    cb_vectors[pid] = {vocab_idx: tfidf_val} (đã L2-normalize)
    
    Feature matrix shape: (n_products, vocab_size)
    Mỗi dòng là TF-IDF vector đã normalize của sản phẩm.
    
    Tham số:
        cb_vectors: dict[int, dict[int, float]] — CB vectors
        n_products: int — tổng số sản phẩm (max_id + 1)
    
    Trả về:
        csr_matrix (n_products × vocab_size) — sparse feature matrix
    """
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
    
    del rows, cols, vals
    gc.collect()
    return feat


def build_cb_similarity(cb_feat, top_k=100, chunk=2000):
    """
    Tính CB similarity matrix = feat @ feat.T (cosine similarity vì đã L2-norm).
    
    Dùng chunked computation để tránh full dense matrix.
    Chỉ giữ top-K tương tự mỗi dòng (ngưỡng > thresh có thể lọc sau).
    
    Tham số:
        cb_feat: csr_matrix — feature matrix (n_products × vocab)
        top_k: int — giữ tối đa top-K mỗi dòng
        chunk: int — kích thước chunk
    
    Trả về:
        csr_matrix (n_products × n_products) — CB similarity matrix
    """
    n = cb_feat.shape[0]
    if n == 0 or cb_feat.nnz == 0:
        return csr_matrix((n, n), dtype=np.float32)
    
    print("  [Hybrid] Computing CB similarity matrix (sparse dot product) ...")
    sim = lil_matrix((n, n), dtype=np.float32)
    
    for start in tqdm(range(0, n, chunk), desc="  CB similarity"):
        end = min(start + chunk, n)
        # chunk_sim = cb_feat[start:end] @ cb_feat.T  → dense (chunk × n)
        chunk_sim = cb_feat[start:end].dot(cb_feat.T)
        
        for i_local, row_idx in enumerate(range(start, end)):
            # Hạn chế: chunk_sim là sparse (ix × n) từ csr @ csr.T
            # Dùng .toarray() → dense để dùng argpartition
            row_dense = chunk_sim[i_local].toarray().ravel()
            row_dense[row_idx] = 0  # Bỏ self-similarity
            if row_dense.size == 0:
                continue

            if top_k < n and row_dense.size > top_k:
                idx = np.argpartition(row_dense, -top_k)[-top_k:]
                vals = row_dense[idx]
            else:
                idx = np.arange(row_dense.size, dtype=np.int32)
                vals = row_dense

            # Lọc positive values
            pos = vals > 0
            pos_idx = idx[pos] if idx.dtype == np.int32 else idx[pos]
            if np.any(pos):
                sim[row_idx, pos_idx] = vals[pos].astype(np.float32)
    
    csr = sim.tocsr()
    del sim; gc.collect()
    print(f"  [Hybrid] CB similarity: {csr.nnz:,} entries")
    return csr


def build_hybrid(confidence, kg_sim, cb_vectors, alpha=HYBRID_ALPHA, beta=HYBRID_BETA, cb_thresh=HYBRID_CB_THRESH):
    """
    Kết hợp Confidence và KG thành hybrid score, loại substitute bằng CB filter.
    
    Quy trình:
        1. Normalize Confidence về [0, 1]
        2. Compute combined scores: α * Conf_norm + β * KG_sim (sparse addition)
        3. Precompute CB similarity matrix sparse từ cb_vectors
        4. Với mỗi dòng, zero-out entries có CB similarity > threshold
        5. Giữ top scoring entries
    
    Tham số:
        confidence: csr_matrix — ma trận Confidence từ build_confidence (unified scoring)
        kg_sim: csr_matrix — ma trận KG similarity từ build_knowledge_graph
        cb_vectors: dict — CB vectors từ build_cb
        alpha: float — trọng số Confidence
        beta: float — trọng số KG
        cb_thresh: float — ngưỡng CB filter (mặc định: 0.8)
        
    Trả về:
        csr_matrix (n_products x n_products) — ma trận hybrid score
    """
    print(f"\n  [Hybrid] α={alpha}, β={beta}, cb_thresh={cb_thresh} ...")
    n = confidence.shape[0]
    
    # Bước 1: Normalize Confidence về [0, 1]
    conf_data = confidence.data.copy()
    max_s = conf_data.max()
    if max_s > 0:
        conf_data /= max_s
    conf_norm = csr_matrix((conf_data, confidence.indices, confidence.indptr), shape=confidence.shape)
    
    # Bước 2: Combine Confidence + KG (sparse addition với trọng số)
    print("  [Hybrid] Combining Confidence + KG scores ...")
    combined = alpha * conf_norm + beta * kg_sim
    del conf_norm
    gc.collect()
    print(f"  [Hybrid] Combined: {combined.nnz:,} entries")
    
    if combined.nnz == 0:
        print("  [Hybrid] Warning: combined matrix empty!")
        return combined
    
    # Bước 3: Precompute CB similarity matrix
    # Nếu có CB vectors, xây sparse feature matrix và similarity
    cb_sim = None
    if cb_vectors and cb_thresh < 1.0:
        cb_feat = build_cb_sparse(cb_vectors, n)
        if cb_feat.nnz > 0:
            cb_sim = build_cb_similarity(cb_feat, top_k=200)
            del cb_feat
            gc.collect()
    
    # Bước 4: CB filter — sử dụng sparse matrix mask
    if cb_sim is not None and cb_sim.nnz > 0:
        print(f"  [Hybrid] Applying CB filter (threshold={cb_thresh}) ...")
        
        # Lọc: với mỗi cặp (i,j), nếu CB_sim > threshold → zero-out trong combined
        # Làm row-by-row để kiểm soát bộ nhớ
        hybrid = combined.tolil()
        cb_coo = cb_sim.tocoo()
        
        # Duyệt qua các cặp CB similarity > threshold
        rows_over = []
        cols_over = []
        row_set = {}
        for i, j, v in tqdm(zip(cb_coo.row, cb_coo.col, cb_coo.data), 
                            desc="  CB filter", total=cb_coo.nnz):
            if v > cb_thresh and i < n and j < n:
                rows_over.append(i)
                cols_over.append(j)
                if i not in row_set:
                    row_set[i] = set()
                row_set[i].add(j)
        
        # Zero-out trong hybrid cho các cặp substitute
        for i in tqdm(row_set, desc="  Removing substitutes"):
            for j in row_set[i]:
                hybrid[i, j] = 0.0
        
        del cb_sim, rows_over, cols_over, row_set
        gc.collect()
        
        # Remove zero rows/cols
        # Giữ nguyên shape, chỉ set 0
        csr_result = hybrid.tocsr()
        del hybrid
        gc.collect()
    else:
        csr_result = combined
    
    # Cleanup
    del combined
    gc.collect()
    print(f"  [Hybrid] Done: {csr_result.nnz:,} entries")
    return csr_result


def save(matrix):
    """Lưu hybrid matrix ra file .npz"""
    save_npz(HYBRID_FILE, matrix)
    print(f"  [Hybrid] Saved: {HYBRID_FILE}")


def load():
    """Tải hybrid matrix từ file .npz"""
    return load_npz(HYBRID_FILE)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(MODELS_DIR.parent))
    from src.features.build_cb import load as load_cb
    
    try:
        confidence = load_npz(CONF_FILE)
        print("  [Hybrid] Loaded Confidence matrix")
    except FileNotFoundError:
        print("  [Hybrid] ERROR: Confidence matrix not found! Run build_confidence.py first.")
        raise
    
    kg_sim = load_npz(MODELS_DIR / "kg_similarity.npz")
    load_cb()  # Nạp cb_vectors vào bộ nhớ
    
    from src.features.build_cb import cb_vectors
    h = build_hybrid(confidence, kg_sim, cb_vectors)
    save(h)
