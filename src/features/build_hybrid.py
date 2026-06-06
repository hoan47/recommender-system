"""
Hybrid — kết hợp SPMI + Knowledge Graph, loại substitute bằng CB

Công thức:
  final_score(A → B) = α * SPMI_norm(A,B) + β * KG_sim(A,B)
  Nếu CB_sim(A,B) > threshold → final_score = 0 (loại substitute)

Trong đó:
  - α: trọng số cho SPMI (càng cao → càng ưu tiên mua kèm)
  - β: trọng số cho KG (càng cao → càng ưu tiên liên quan qua đồ thị)
  - threshold: ngưỡng CB similarity (nếu 2 sản phẩm quá giống → loại)

SPMI và KG đều tìm complementary (mua kèm) nhưng theo cách khác nhau.
CB filter bỏ đi các substitute (sản phẩm thay thế) vì không nên recommend
sản phẩm quá giống với sản phẩm đang xem.
"""

import gc
import numpy as np
from scipy.sparse import load_npz, save_npz, csr_matrix, lil_matrix
from tqdm import tqdm

from src.config import MODELS_DIR
from src.config import HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH

# File lưu hybrid matrix
HYBRID_FILE = MODELS_DIR / "hybrid_matrix.npz"

def build_hybrid(spmi, kg_sim, cb_vectors, alpha=HYBRID_ALPHA, beta=HYBRID_BETA, cb_thresh=HYBRID_CB_THRESH):
    """
    Kết hợp SPMI và KG thành hybrid score, loại substitute bằng CB filter.
    
    Quy trình:
        1. Normalize SPMI về [0, 1] (vì SPMI có thể > 1, trong khi KG similarity nằm trong [0, 1])
        2. Với mỗi sản phẩm i:
            - Lấy SPMI scores → nhân với α
            - Lấy KG scores → nhân với β
            - Cộng dồn vào candidate dict
            - Với mỗi candidate j, tính CB similarity(i, j)
            - Nếu CB similarity > threshold → j là substitute → loại
            - Sắp xếp candidates theo score giảm dần
    
    Tham số:
        spmi: csr_matrix — ma trận SPMI từ build_spmi
        kg_sim: csr_matrix — ma trận KG similarity từ build_knowledge_graph
        cb_vectors: dict — CB vectors từ build_cb (dùng để tính similarity on-the-fly)
        alpha: float — trọng số SPMI (mặc định: 0.6)
        beta: float — trọng số KG (mặc định: 0.4)
        cb_thresh: float — ngưỡng CB filter (mặc định: 0.8)
        
    Trả về:
        csr_matrix (n_products x n_products) — ma trận hybrid score
    """
    print(f"\n  [Hybrid] α={alpha}, β={beta}, cb_thresh={cb_thresh} ...")
    n = spmi.shape[0]

    # Bước 1: Normalize SPMI về [0, 1]
    # SPMI values có thể > 1 (do log scale), cần đưa về cùng thang với KG similarity
    spmi_data = spmi.data.copy()
    max_s = spmi_data.max()
    if max_s > 0:
        spmi_data /= max_s  # Chia cho max → [0, 1]
    spmi_norm = csr_matrix((spmi_data, spmi.indices, spmi.indptr), shape=spmi.shape)

    # Bước 2: Kết hợp scores và filter
    hybrid = lil_matrix((n, n), dtype=np.float32)

    for i in tqdm(range(n), desc="  Hybrid"):
        # Gom scores từ SPMI và KG
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

        # CB filter: kiểm tra nếu sản phẩm j quá giống i → loại
        cb_vec_i = cb_vectors.get(i, {})
        filtered = {}
        for j, s in scores.items():
            if s <= 0:
                continue
            # Tính CB similarity giữa i và j
            cb_vec_j = cb_vectors.get(j, {})
            if cb_vec_i and cb_vec_j:
                # Dot product trên key chung (đã L2-normalize nên dot = cosine)
                if len(cb_vec_i) > len(cb_vec_j):
                    va, vb = cb_vec_j, cb_vec_i
                else:
                    va, vb = cb_vec_i, cb_vec_j
                cb_sim = sum(va[t] * vb[t] for t in va if t in vb)
                if cb_sim > cb_thresh:
                    continue  # Quá giống → substitute → loại bỏ
            filtered[j] = s

        if filtered:
            # Sắp xếp giảm dần theo score
            items = list(filtered.items())
            items.sort(key=lambda x: -x[1])
            hybrid[i, [j for j, _ in items]] = [s for _, s in items]

    csr = hybrid.tocsr()
    del hybrid; gc.collect()
    print(f"  [Hybrid] Done: {csr.nnz:,} entries")
    return csr

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
    
    # Tải SPMI, KG, CB từ các bước trước
    spmi = load_npz(MODELS_DIR / "spmi_matrix.npz")
    kg_sim = load_npz(MODELS_DIR / "kg_similarity.npz")
    load_cb()  # Nạp cb_vectors vào bộ nhớ
    
    from src.features.build_cb import cb_vectors
    h = build_hybrid(spmi, kg_sim, cb_vectors)
    save(h)