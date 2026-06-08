"""
Unified Scoring — Confidence Item-based Collaborative Filtering

Công thức unified scoring (asymmetric, directed graph):
  ochiai(A,B) = cnt / sqrt(freq[A] * freq[B])          # Cosine Similarity
  log_ab      = log1p(cnt)                               # Popularity Bonus (log)
  conf(A→B)   = cnt / freq[A]                            # Conditional Probability
  score(A→B)  = ochiai(A,B) * conf(A→B) * log_ab        # Unified score

  Reorder bonus (optional):
    rr_bonus = 1.0 + REORDER_BONUS * avg(reorder_rate[A], reorder_rate[B])
    score *= rr_bonus

Ma trận lưu dạng scipy.sparse.csr_matrix để tiết kiệm bộ nhớ và tương thích với các module khác.

Co-occurrence dùng Numba JIT + typed Dict → ~100x nhanh hơn Python loop.
Scoring dùng Python loop có numpy argpartition cho top-K selection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from numba import njit
from numba.typed import Dict
from numba.core import types as nb_types

from src.config import MODELS_DIR, CONF_FREQ_MIN, CONF_TOP_K, REORDER_BONUS

# File lưu ma trận co-occurrence và Confidence (unified scoring)
COOC_FILE = MODELS_DIR / "cooc_matrix.npz"
CONF_FILE = MODELS_DIR / "confidence_matrix.npz"


# ============================================================
# Numba JIT — Co-occurrence counting
# ============================================================

@njit
def _cooc_from_orders(order_ids, product_ids, n_products):
    """
    Đếm co-occurrence dùng Numba JIT + typed Dict (accumulate in-place).

    Tham số:
        order_ids: ndarray(int32) — order_id đã sort ổn định
        product_ids: ndarray(int32) — product_id tương ứng
        n_products: int — số sản phẩm (max_id + 1)

    Trả về:
        rows: ndarray(int32) — row indices cho COO matrix (tam giác trên i < j)
        cols: ndarray(int32) — col indices cho COO matrix
        vals: ndarray(float64) — values (co-occurrence count)
        freq: ndarray(float64) — document frequency mỗi sản phẩm
    """
    # Dictionary: key = i * n_products + j → value = co-occurrence count
    cooc_dict = Dict.empty(
        key_type=nb_types.int64,
        value_type=nb_types.float64,
    )
    freq = np.zeros(n_products, dtype=np.float64)

    n = len(order_ids)
    start = 0
    while start < n:
        # Tìm boundary của order hiện tại
        end = start
        oid = order_ids[start]
        while end < n and order_ids[end] == oid:
            end += 1

        # Lấy unique products trong order
        local = product_ids[start:end]
        local_sorted = np.sort(local)
        if len(local_sorted) > 0:
            uniq_mask = np.ones(len(local_sorted), dtype=np.bool_)
            for k in range(1, len(local_sorted)):
                if local_sorted[k] == local_sorted[k - 1]:
                    uniq_mask[k] = False
            uniq = local_sorted[uniq_mask]
        else:
            uniq = local_sorted

        k = len(uniq)
        if k < 2:
            for idx_u in range(k):
                freq[int(uniq[idx_u])] += 1.0
            start = end
            continue

        for idx_u in range(k):
            freq[int(uniq[idx_u])] += 1.0

        for ai in range(k):
            for bi in range(ai + 1, k):
                a = int(uniq[ai])
                b = int(uniq[bi])
                key = a * n_products + b
                cooc_dict[key] = cooc_dict.get(key, 0.0) + 1.0

        start = end

    # Convert Dict → COO arrays (tam giác trên i < j)
    n_keys = len(cooc_dict)
    rows = np.empty(n_keys, dtype=np.int32)
    cols = np.empty(n_keys, dtype=np.int32)
    vals = np.empty(n_keys, dtype=np.float64)

    idx = 0
    for key, val in cooc_dict.items():
        rows[idx] = key // n_products
        cols[idx] = key % n_products
        vals[idx] = val
        idx += 1

    return rows, cols, vals, freq


# ============================================================
# Build co-occurrence matrix
# ============================================================

def build_cooc(prior_df):
    """
    Đếm co-occurrence từ prior orders dùng Numba JIT accelerated.

    Tham số:
        prior_df: DataFrame từ load_prior() — cần cột order_id, product_id

    Trả về:
        cooc_csr: csr_matrix (n_products x n_products) — ma trận co-occurrence đối xứng
        freq: numpy array (n_products,) — số đơn hàng chứa mỗi sản phẩm
    """
    print("\n  [Confidence] Building co-occurrence (Numba JIT) ...")

    n = int(prior_df["product_id"].max()) + 1

    sorted_df = prior_df.sort_values("order_id")
    order_ids = sorted_df["order_id"].values.astype(np.int32)
    product_ids = sorted_df["product_id"].values.astype(np.int32)

    print(f"  [Confidence] Orders: {len(prior_df['order_id'].unique()):,}, "
          f"Records: {len(prior_df):,}, Products: {n:,}")

    rows, cols, vals, freq = _cooc_from_orders(order_ids, product_ids, n)

    cooc_triu = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    cooc = cooc_triu + cooc_triu.T
    cooc_csr = cooc.tocsr()

    del sorted_df, order_ids, product_ids, rows, cols, vals, cooc_triu, cooc
    gc.collect()

    print(f"  [Confidence] Co-occurrence: {cooc_csr.nnz:,} entries, "
          f"density={cooc_csr.nnz / (n * n) * 100:.2f}%")
    return cooc_csr, freq


# ============================================================
# Build Confidence (Unified Scoring)
# ============================================================

def build_confidence(cooc_csr, freq, reorder_rate=None,
                     freq_min=CONF_FREQ_MIN, top_k=CONF_TOP_K):
    """
    Tính Confidence matrix = unified scoring + reorder bonus, bất đối xứng.

    Công thức mỗi cặp (i→j):
      cnt = cooc[i, j]
      ochiai = cnt / sqrt(freq[i] * freq[j])
      log_ab = log1p(cnt)
      conf   = cnt / freq[i]
      score  = ochiai * conf * log_ab * rr_bonus

    Reorder bonus:
      rr_bonus = 1.0 + REORDER_BONUS * avg(reorder_rate[i], reorder_rate[j])

    Tham số:
        cooc_csr: csr_matrix — ma trận co-occurrence đối xứng
        freq: numpy array — order frequency mỗi sản phẩm
        reorder_rate: dict[int, float] — product_id → reorder rate (nếu None, rr_bonus = 1)
        freq_min: int — ngưỡng tối thiểu freq(i) để recommend
        top_k: int — số lượng gợi ý tối đa mỗi sản phẩm

    Trả về:
        confidence: csr_matrix (n_products x n_products) — KHÔNG đối xứng
    """
    n = cooc_csr.shape[0]
    print(f"\n  [Confidence] Computing unified score (freq_min={freq_min}, top_k={top_k}) ...")

    rows_list = []
    cols_list = []
    vals_list = []
    nnz_total = 0

    for i in range(n):
        fi = freq[i]
        if fi < freq_min:
            continue

        row_start = cooc_csr.indptr[i]
        row_end = cooc_csr.indptr[i + 1]
        if row_start == row_end:
            continue

        local_j = cooc_csr.indices[row_start:row_end]
        local_c = cooc_csr.data[row_start:row_end]

        m = len(local_j)
        if m == 0:
            continue

        scores = np.zeros(m, dtype=np.float64)
        for idx in range(m):
            j = local_j[idx]
            cnt = local_c[idx]
            fj = freq[j]
            if fj == 0:
                continue

            ochiai = cnt / math.sqrt(fi * fj)
            log_ab = math.log1p(cnt)
            conf_ij = cnt / fi
            score = ochiai * conf_ij * log_ab

            if reorder_rate is not None:
                rr_i = reorder_rate.get(i, 0.5)
                rr_j = reorder_rate.get(j, 0.5)
                rr_bonus = 1.0 + REORDER_BONUS * (rr_i + rr_j) / 2.0
                score *= rr_bonus

            scores[idx] = score

        if np.all(scores == 0):
            continue

        n_keep = min(m, top_k)
        if n_keep < m:
            top_idx = np.argpartition(scores, -n_keep)[-n_keep:]
            top_sorted = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_sorted = np.argsort(-scores)

        for rank in range(n_keep):
            idx = top_sorted[rank]
            val = scores[idx]
            if val > 0:
                rows_list.append(i)
                cols_list.append(local_j[idx])
                vals_list.append(val)
                nnz_total += 1

    if nnz_total > 0:
        confidence = csr_matrix(
            (np.array(vals_list, dtype=np.float32),
             (np.array(rows_list, dtype=np.int32),
              np.array(cols_list, dtype=np.int32))),
            shape=(n, n),
            dtype=np.float32,
        )
    else:
        confidence = csr_matrix((n, n), dtype=np.float32)

    del rows_list, cols_list, vals_list
    gc.collect()

    print(f"  [Confidence] Done: {confidence.nnz:,} entries, "
          f"density={confidence.nnz / (n * n) * 100:.4f}%")
    return confidence


# ============================================================
# Save / Load
# ============================================================

def save(cooc, conf):
    """Lưu ma trận co-occurrence và Confidence ra file .npz"""
    save_npz(COOC_FILE, cooc)
    save_npz(CONF_FILE, conf)
    print(f"  [Confidence] Saved: {COOC_FILE}, {CONF_FILE}")


def load():
    """Tải ma trận co-occurrence và Confidence từ file .npz"""
    return load_npz(COOC_FILE), load_npz(CONF_FILE)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    from src.data_loader import load_prior, load_products

    prior = load_prior()
    products_df = load_products()

    # Tính reorder rate để có bonus khi build confidence
    reorder_rate = prior.groupby('product_id')['reordered'].mean().to_dict()

    cooc, freq = build_cooc(prior)
    conf = build_confidence(cooc, freq, reorder_rate=reorder_rate)
    save(cooc, conf)