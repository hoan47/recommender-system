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
"""

import gc
import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from tqdm import tqdm

from src.config import MODELS_DIR, SPMI_K, TOTAL_PRIOR_ORDERS, SPMI_TOP_K

# File lưu ma trận co-occurrence và SPMI (sparse .npz)
COOC_FILE = MODELS_DIR / "cooc_matrix.npz"
SPMI_FILE = MODELS_DIR / "spmi_matrix.npz"

def build_cooc(prior_df):
    """
    Đếm co-occurrence từ prior orders.
    Với mỗi đơn hàng (group theo order_id), mỗi cặp không thứ tự (A,B) được
    tính 1 lần cho cả cooc[A][B] và cooc[B][A] (ma trận đối xứng).
    
    Tham số:
        prior_df: DataFrame từ load_prior() — cần cột order_id, product_id
    
    Trả về:
        cooc_csr: csr_matrix (n_products x n_products) — ma trận co-occurrence
        freq: numpy array (n_products,) — số đơn hàng chứa mỗi sản phẩm
    """
    print("\n  [SPMI] Building co-occurrence ...")
    n = prior_df["product_id"].max() + 1  # Số sản phẩm = max_id + 1
    grouped = prior_df.groupby("order_id")  # Gom nhóm theo đơn hàng
    # LIL matrix cho phép increment từng phần tử hiệu quả
    cooc = lil_matrix((n, n), dtype=np.float64)
    freq = np.zeros(n, dtype=np.float64)  # Document frequency cho mỗi sản phẩm

    for _, grp in tqdm(grouped, desc="  Co-occurrence"):
        prods = grp["product_id"].values
        if len(prods) < 2:
            continue  # Đơn 1 sản phẩm không có cặp
        # Cập nhật order frequency
        uniq = np.unique(prods)
        freq[uniq] += 1
        # Đếm tất cả cặp không thứ tự (O(n^2) nhưng đơn hàng trung bình nhỏ)
        for i in range(len(prods)):
            for j in range(i + 1, len(prods)):
                a, b = int(prods[i]), int(prods[j])
                cooc[a, b] += 1
                cooc[b, a] += 1  # Ma trận đối xứng

    cooc_csr = cooc.tocsr()  # Chuyển sang CSR (tiết kiệm bộ nhớ hơn)
    del cooc; gc.collect()
    print(f"  [SPMI] Co-occurrence: {cooc_csr.nnz:,} entries")
    return cooc_csr, freq

def build_spmi(cooc_csr, freq, k=SPMI_K, top_k=SPMI_TOP_K):
    """
    Tính SPMI từ ma trận co-occurrence.
    
    Công thức: SPMI(A,B) = max(log(cooc * N / (freq_i * freq_j)) - log(k), 0)
    SPMI càng cao → A và B càng có xu hướng mua kèm.
    SPMI = 0 → A và B độc lập hoặc conflict.
    
    Tham số:
        cooc_csr: csr_matrix — ma trận co-occurrence
        freq: numpy array — order frequency
        k: int — threshold shift (mặc định: SPMI_K=3)
        top_k: int — chỉ giữ top-K mỗi dòng
    
    Trả về:
        spmi: csr_matrix (n_products x n_products) — ma trận SPMI
    """
    print(f"\n  [SPMI] Computing SPMI k={k} ...")
    n = cooc_csr.shape[0]
    log_shift = math.log(k)  # Giá trị shift: log(k)
    rows, cols, vals = [], [], []

    for i in tqdm(range(n), desc="  SPMI"):
        row = cooc_csr[i]
        if row.nnz == 0:
            continue
        fi = freq[i]
        if fi == 0:
            continue  # Sản phẩm không xuất hiện trong prior
        # Tính SPMI cho tất cả cặp của sản phẩm i
        scores = []
        for j, c in zip(row.indices, row.data):
            fj = freq[j]
            if fj == 0:
                continue
            # Công thức PMI chuẩn
            pmi = math.log(c * TOTAL_PRIOR_ORDERS / (fi * fj))
            spmi = max(pmi - log_shift, 0)  # Shift + cắt về 0
            if spmi > 0:
                scores.append((j, spmi))
        if not scores:
            continue
        # Chỉ giữ top-K sản phẩm mua kèm mạnh nhất
        scores.sort(key=lambda x: -x[1])
        for j, s in scores[:top_k]:
            rows.append(i)
            cols.append(j)
            vals.append(s)

    # Xây dựng CSR matrix từ danh sách (rows, cols, vals)
    spmi = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    del rows, cols, vals; gc.collect()
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