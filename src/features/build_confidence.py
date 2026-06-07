"""
Confidence — Item-based Collaborative Filtering dùng Conditional Probability

Công thức:
  Confidence(A → B) = cooc(A,B) / freq(A)

Đọc: "Nếu mua A, xác suất cũng mua B là bao nhiêu %"
Đây là câu trả lời trực tiếp cho bài toán recommend "mua kèm".

Đặc điểm:
  - KHÔNG đối xứng: Confidence(A→B) ≠ Confidence(B→A)
  - Chỉ recommend từ product có freq(A) >= FREQ_MIN (loại nhiễu từ sản phẩm quá hiếm)
  - Không có threshold ảo (không log, không shift) — score là % thực tế
  - Giữ top-K mỗi dòng để giảm dung lượng

Reuses:
  - _cooc_from_orders() từ build_spmi (Numba JIT)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from tqdm import tqdm

from src.config import MODELS_DIR, CONF_FREQ_MIN, CONF_TOP_K
from src.features.build_spmi import _cooc_from_orders

# File lưu ma trận co-occurrence và Confidence
COOC_FILE = MODELS_DIR / "cooc_matrix.npz"
CONF_FILE = MODELS_DIR / "confidence_matrix.npz"


def build_cooc(prior_df):
    """
    Đếm co-occurrence từ prior orders (giống build_spmi, để convenience).

    Tham số:
        prior_df: DataFrame từ load_prior()

    Trả về:
        cooc_csr: csr_matrix (n_products x n_products) — ma trận co-occurrence đối xứng
        freq: numpy array (n_products,) — số đơn hàng chứa mỗi sản phẩm
    """
    print("\n  [Confidence] Building co-occurrence (Numba JIT) ...")

    n = int(prior_df["product_id"].max()) + 1
    sorted_df = prior_df.sort_values("order_id")
    order_ids = sorted_df["order_id"].values.astype(np.int32)
    product_ids = sorted_df["product_id"].values.astype(np.int32)

    print(f"  [Confidence]   Orders: {len(prior_df['order_id'].unique()):,}, "
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


def build_confidence(cooc_csr, freq, freq_min=CONF_FREQ_MIN, top_k=CONF_TOP_K):
    """
    Tính Confidence matrix = cooc(i,j) / freq(i), không đối xứng.

    Chỉ recommend từ product A nếu freq(A) >= freq_min (loại sản phẩm quá hiếm).
    Giữ top-K mỗi dòng.

    Tham số:
        cooc_csr: csr_matrix — ma trận co-occurrence đối xứng
        freq: numpy array — order frequency mỗi sản phẩm
        freq_min: int — ngưỡng tối thiểu freq(A) để recommend
        top_k: int — số lượng gợi ý tối đa mỗi sản phẩm

    Trả về:
        confidence: csr_matrix (n_products x n_products) — KHÔNG đối xứng
    """
    n = cooc_csr.shape[0]
    print(f"\n  [Confidence] Computing confidence (freq_min={freq_min}, top_k={top_k}) ...")

    # Dùng LIL matrix để xây dựng từng dòng dễ dàng
    conf = csr_matrix((n, n), dtype=np.float32)

    rows_list = []
    cols_list = []
    vals_list = []
    nnz_total = 0

    for i in range(n):
        fi = freq[i]
        if fi < freq_min:
            continue  # Bỏ qua sản phẩm quá hiếm

        row_start = cooc_csr.indptr[i]
        row_end = cooc_csr.indptr[i + 1]
        if row_start == row_end:
            continue

        # Thu thập tất cả (j, confidence)
        local_j = cooc_csr.indices[row_start:row_end]
        local_c = cooc_csr.data[row_start:row_end].copy()

        # Confidence = cooc / freq(A)
        local_conf = local_c / fi

        if len(local_conf) == 0:
            continue

        # Sort giảm dần theo confidence, giữ top_k
        n_local = len(local_conf)
        n_keep = min(n_local, top_k)

        if n_keep < n_local:
            idx = np.argpartition(local_conf, -n_keep)[-n_keep:]
            top_sorted = idx[np.argsort(-local_conf[idx])]
        else:
            top_sorted = np.argsort(-local_conf)

        for rank in range(n_keep):
            j = local_j[top_sorted[rank]]
            val = local_conf[top_sorted[rank]]
            if val > 0:
                rows_list.append(i)
                cols_list.append(j)
                vals_list.append(val)
                nnz_total += 1

    if nnz_total > 0:
        conf = csr_matrix(
            (np.array(vals_list, dtype=np.float32),
             (np.array(rows_list, dtype=np.int32),
              np.array(cols_list, dtype=np.int32))),
            shape=(n, n),
            dtype=np.float32,
        )

    del rows_list, cols_list, vals_list
    gc.collect()

    print(f"  [Confidence] Done: {conf.nnz:,} entries, "
          f"density={conf.nnz / (n * n) * 100:.4f}%")
    return conf


def save(cooc, conf):
    """Lưu ma trận co-occurrence và Confidence ra file .npz"""
    save_npz(COOC_FILE, cooc)
    save_npz(CONF_FILE, conf)
    print(f"  [Confidence] Saved: {COOC_FILE}, {CONF_FILE}")


def load():
    """Tải ma trận co-occurrence và Confidence từ file .npz"""
    return load_npz(COOC_FILE), load_npz(CONF_FILE)


if __name__ == "__main__":
    from src.data_loader import load_prior
    prior = load_prior()
    cooc, freq = build_cooc(prior)
    conf = build_confidence(cooc, freq)
    save(cooc, conf)