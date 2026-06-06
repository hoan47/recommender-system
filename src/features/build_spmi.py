"""
SPMI — Shifted Positive PMI từ co-occurrence matrix

Tìm sản phẩm MUA KÈM (complementary) dựa trên lịch sử mua hàng.
Công thức:
  PMI(A,B) = log(cooc[A][B] * N / (freq[A] * freq[B]))
  SPMI(A,B) = max(PMI(A,B) - log(k), 0)

Trong đó:
  - cooc[A][B] = số đơn hàng có cả A và B
  - freq[A] = số đơn hàng có A (document frequency)
  - N = tổng số đơn hàng trong prior
  - k = threshold shift (càng cao → càng loại bỏ nhiều cặp yếu)

SPMI > 0 nghĩa là A và B có mối quan hệ mua kèm thực sự.
Chỉ giữ top-K mỗi dòng để giảm dung lượng ma trận.

Optimizations:
  1. Co-occurrence dùng Numba JIT + typed Dict → ~100x nhanh hơn Python loop
  2. SPMI computation dùng vectorized sparse + argpartition thay vì sort từng dòng
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
from tqdm import tqdm

from src.config import MODELS_DIR, SPMI_K, TOTAL_PRIOR_ORDERS, SPMI_TOP_K

# File lưu ma trận co-occurrence và SPMI (sparse .npz)
COOC_FILE = MODELS_DIR / "cooc_matrix.npz"
SPMI_FILE = MODELS_DIR / "spmi_matrix.npz"


@njit
def _cooc_from_orders(order_ids, product_ids, n_products):
    """
    Đếm co-occurrence dùng Numba JIT + typed Dict (accumulate in-place).
    
    Tham số:
        order_ids: ndarray(int32) — order_id đã sort ổn định
        product_ids: ndarray(int32) — product_id tương ứng
        n_products: int — số sản phẩm (max_id + 1)
    
    Trả về:
        rows: ndarray(int32) — row indices cho COO matrix
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
        # Mask để loại bỏ duplicates
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
            # Cập nhật freq kể cả khi chỉ có 1 sản phẩm
            for idx_u in range(k):
                freq[int(uniq[idx_u])] += 1.0
            start = end
            continue
        
        # Cập nhật order frequency cho mỗi sản phẩm
        for idx_u in range(k):
            freq[int(uniq[idx_u])] += 1.0
        
        # Đếm tất cả cặp không thứ tự
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


def build_cooc(prior_df):
    """
    Đếm co-occurrence từ prior orders dùng Numba JIT accelerated.
    
    Tham số:
        prior_df: DataFrame từ load_prior() — cần cột order_id, product_id
    
    Trả về:
        cooc_csr: csr_matrix (n_products x n_products) — ma trận co-occurrence đối xứng
        freq: numpy array (n_products,) — số đơn hàng chứa mỗi sản phẩm
    """
    print("\n  [SPMI] Building co-occurrence (Numba JIT) ...")
    
    n = int(prior_df["product_id"].max()) + 1
    
    # Sort theo order_id để Numba JIT dễ dàng scan
    sorted_df = prior_df.sort_values("order_id")
    order_ids = sorted_df["order_id"].values.astype(np.int32)
    product_ids = sorted_df["product_id"].values.astype(np.int32)
    
    print(f"  [SPMI]   Orders: {len(prior_df['order_id'].unique()):,}, "
          f"Records: {len(prior_df):,}, Products: {n:,}")
    
    # Gọi Numba JIT function
    rows, cols, vals, freq = _cooc_from_orders(order_ids, product_ids, n)
    
    # Xây dựng COO matrix (tam giác trên)
    cooc_triu = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    
    # Symmetrize: cooc = triu + triu.T (ma trận đối xứng)
    cooc = cooc_triu + cooc_triu.T
    cooc_csr = cooc.tocsr()
    
    # Dọn dẹp
    del sorted_df, order_ids, product_ids, rows, cols, vals, cooc_triu, cooc
    gc.collect()
    
    print(f"  [SPMI] Co-occurrence: {cooc_csr.nnz:,} entries, "
          f"density={cooc_csr.nnz / (n * n) * 100:.2f}%")
    return cooc_csr, freq


@njit
def _spmi_rowwise(n, cooc_indptr, cooc_indices, cooc_data, freq, n_orders, log_shift, top_k):
    """
    Tính SPMI row-by-row và collect top-K entries, dùng Numba JIT.
    
    Thao tác trực tiếp trên CSR internal arrays để tránh Python overhead.
    Mỗi dòng: tính SPMI cho tất cả cặp, giữ top-K, lưu vào COO arrays.
    
    Tham số:
        n: int — số sản phẩm
        cooc_indptr, cooc_indices, cooc_data: CSR internal arrays của cooc matrix
        freq: ndarray — document frequency
        n_orders: int — tổng số đơn hàng (TOTAL_PRIOR_ORDERS)
        log_shift: float — log(SPMI_K)
        top_k: int — số lượng top entries giữ mỗi dòng
    
    Trả về:
        rows, cols, vals: 1D arrays cho COO sparse matrix
    """
    log_n = math.log(n_orders)
    
    # Pass 1: Đếm tổng số entries sau top-K selection
    total_entries = 0
    for i in range(n):
        row_start = cooc_indptr[i]
        row_end = cooc_indptr[i + 1]
        if row_start == row_end:
            continue
        fi = freq[i]
        if fi == 0:
            continue
        
        # Count entries with SPMI > 0
        n_valid = 0
        for j_idx in range(row_start, row_end):
            j = cooc_indices[j_idx]
            c = cooc_data[j_idx]
            fj = freq[j]
            if fj == 0:
                continue
            pmi = math.log(c) + log_n - math.log(fi) - math.log(fj)
            spmi_val = pmi - log_shift
            if spmi_val > 0:
                n_valid += 1
        
        if n_valid == 0:
            continue
        # Chỉ giữ top_k
        total_entries += min(n_valid, top_k)
    
    rows = np.empty(total_entries, dtype=np.int32)
    cols = np.empty(total_entries, dtype=np.int32)
    vals = np.empty(total_entries, dtype=np.float64)
    
    # Pass 2: Fill
    out_idx = 0
    for i in range(n):
        row_start = cooc_indptr[i]
        row_end = cooc_indptr[i + 1]
        if row_start == row_end:
            continue
        fi = freq[i]
        if fi == 0:
            continue
        
        # Thu thập tất cả (j, spmi) với spmi > 0
        # Dùng local arrays vì Numba không hỗ trợ list of tuples
        local_j = np.empty(row_end - row_start, dtype=np.int32)
        local_spmi = np.empty(row_end - row_start, dtype=np.float64)
        n_local = 0
        
        for j_idx in range(row_start, row_end):
            j = cooc_indices[j_idx]
            c = cooc_data[j_idx]
            fj = freq[j]
            if fj == 0:
                continue
            pmi = math.log(c) + log_n - math.log(fi) - math.log(fj)
            spmi_val = pmi - log_shift
            if spmi_val > 0:
                local_j[n_local] = j
                local_spmi[n_local] = spmi_val
                n_local += 1
        
        if n_local == 0:
            continue
        
        # Sort giảm dần theo spmi value dùng argpartition cho top_k
        n_keep = min(n_local, top_k)
        # Numba support np.argpartition
        if n_keep < n_local:
            # Chỉ lấy top_k bằng argpartition
            top_idx = np.argpartition(local_spmi[:n_local], -n_keep)[-n_keep:]
            # Sort top_idx giảm dần
            top_sorted = top_idx[np.argsort(-local_spmi[top_idx])]
        else:
            # Nếu ít hơn top_k, sort tất cả giảm dần
            top_sorted = np.argsort(-local_spmi[:n_local])
        
        for rank in range(n_keep):
            idx = top_sorted[rank]
            rows[out_idx] = i
            cols[out_idx] = local_j[idx]
            vals[out_idx] = local_spmi[idx]
            out_idx += 1
    
    return rows, cols, vals


def build_spmi(cooc_csr, freq, k=SPMI_K, top_k=SPMI_TOP_K):
    """
    Tính SPMI từ ma trận co-occurrence dùng Numba JIT accelerated.
    
    Công thức: SPMI(A,B) = max(log(cooc * N / (freq_i * freq_j)) - log(k), 0)
    
    Tham số:
        cooc_csr: csr_matrix — ma trận co-occurrence
        freq: numpy array — order frequency
        k: int — threshold shift
        top_k: int — chỉ giữ top-K mỗi dòng
    
    Trả về:
        spmi: csr_matrix (n_products x n_products) — ma trận SPMI
    """
    print(f"\n  [SPMI] Computing SPMI k={k}, top_k={top_k} (Numba JIT) ...")
    n = cooc_csr.shape[0]
    log_shift = math.log(k)
    
    # Gọi Numba JIT function để tính SPMI và chọn top-K
    rows, cols, vals = _spmi_rowwise(
        n,
        cooc_csr.indptr, cooc_csr.indices, cooc_csr.data,
        freq, float(TOTAL_PRIOR_ORDERS), log_shift, top_k,
    )
    
    # Xây dựng CSR matrix từ COO
    spmi = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    
    del rows, cols, vals
    gc.collect()
    print(f"  [SPMI] Done: {spmi.nnz:,} entries")
    return spmi


def save(cooc, spmi):
    """Lưu ma trận co-occurrence và SPMI ra file .npz"""
    save_npz(COOC_FILE, cooc)
    save_npz(SPMI_FILE, spmi)
    print(f"  [SPMI] Saved: {COOC_FILE}, {SPMI_FILE}")


def load():
    """Tải ma trận co-occurrence và SPMI từ file .npz"""
    return load_npz(COOC_FILE), load_npz(SPMI_FILE)


if __name__ == "__main__":
    from src.data_loader import load_prior
    prior = load_prior()
    cooc, freq = build_cooc(prior)
    spmi = build_spmi(cooc, freq)
    save(cooc, spmi)