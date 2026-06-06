"""
Shifted Positive PMI (SPMI) — Collaborative Filtering model.

Xây dựng co-occurrence item-item từ lịch sử prior, sau đó áp dụng
SPMI để lọc nhiễu và tìm các cặp sản phẩm thực sự mua kèm.

Đếm co-occurrence:
  - Với mỗi đơn hàng trong prior, mỗi cặp sản phẩm (A, B) trong đơn đó
    được tính là một co-occurrence.
  - Xử lý theo chunk để xử lý 32.4M records.

Công thức SPMI:
  PMI(A,B) = log(cooc[A][B] * total_prior_orders / (freq[A] * freq[B]))
  SPMI(A,B) = max(PMI(A,B) - log(k), 0)

  Trong đó freq[A] = số đơn hàng chứa sản phẩm A (document frequency),
  KHÔNG phải tổng lượt mua. total_prior_orders = 3,214,874.

Tuning hyperparameter:
  - Giá trị k: [1, 2, 3, 5, 10]
  - Tune trên tập TRAIN (in-sample leave-one-out)
  - Tập TEST không bao giờ được dùng trong quá trình tune

Phụ thuộc: src.utils.data_loader

Outputs:
  - models/cooc_matrix.npz      - Ma trận co-occurrence sparse gốc
  - models/spmi_matrix.npz      - Ma trận SPMI sparse (chỉ giữ SPMI > 0)
  - models/spmi_best_k.json     - Giá trị k tốt nhất và metrics trên train
"""

import gc
import json
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from tqdm import tqdm

# Thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
#  BƯỚC 1 — Đếm co-occurrence
# =============================================================================

def _build_cooc_lil(grouped, n_products):
    """Duyệt qua từng đơn hàng để đếm co-occurrence và order frequency.

    Dùng LIL matrix để increment hiệu quả.

    Tham số
    ----------
    grouped : pd.DataFrameGroupBy
        Nhóm prior_df theo order_id.
    n_products : int
        Số sản phẩm unique.

    Trả về
    -------
    tuple: (cooc_lil, order_freqs)
        cooc_lil: lil_matrix (n_products × n_products)
        order_freqs: numpy array (n_products,)
    """
    cooc        = lil_matrix((n_products, n_products), dtype=np.float64)
    order_freqs = np.zeros(n_products, dtype=np.float64)

    for _order_id, group in tqdm(grouped, desc="  Đếm co-occurrence", unit="orders"):
        products = group["product_id"].values
        n = len(products)

        # Bỏ qua đơn chỉ có 1 sản phẩm (không có cặp)
        if n < 2:
            continue

        # Cập nhật order frequencies (mỗi sản phẩm trong đơn này → freq++)
        unique_products      = np.unique(products)
        order_freqs[unique_products] += 1

        # Đếm tất cả cặp không thứ tự
        for i in range(n):
            for j in range(i + 1, n):
                a, b = int(products[i]), int(products[j])
                cooc[a, b] += 1
                cooc[b, a] += 1

    return cooc, order_freqs


def count_cooccurrence(prior_df, n_products=None):
    """
    Đếm co-occurrence pairs từ prior orders.

    Với mỗi đơn hàng, mỗi cặp không thứ tự (A, B) tăng
    cooc[A][B] và cooc[B][A] lên 1.

    Tham số
    ----------
    prior_df : pd.DataFrame
        Các cột: order_id, product_id (từ order_products__prior.csv).
    n_products : int hoặc None
        Số sản phẩm unique (nếu None, tự suy từ dữ liệu).

    Trả về
    -------
    tuple: (cooc_matrix, order_freqs)
        cooc_matrix: scipy.sparse.csr_matrix (n_products x n_products)
            cooc[A][B] = số đơn hàng mà cả A và B cùng xuất hiện.
        order_freqs: numpy array (n_products,)
            order_freqs[A] = số đơn hàng chứa sản phẩm A.
    """
    if n_products is None:
        n_products = prior_df["product_id"].max() + 1

    print(f"  Số sản phẩm (không gian index): {n_products:,}")

    n_orders = prior_df["order_id"].nunique()
    print(f"  Đang xử lý {n_orders:,} đơn hàng ({len(prior_df):,} interactions)...")

    grouped     = prior_df.groupby("order_id")
    cooc, order_freqs = _build_cooc_lil(grouped, n_products)

    print(f"  Đã xây dựng co-occurrence matrix. Đang chuyển sang CSR...")
    cooc_csr = cooc.tocsr()

    del cooc; gc.collect()

    print(f"  Non-zero cooc entries: {cooc_csr.nnz:,}")
    print(f"  Trung bình non-zero mỗi sản phẩm: {cooc_csr.nnz / n_products:.1f}")

    return cooc_csr, order_freqs


# =============================================================================
#  BƯỚC 2 — Tính SPMI
# =============================================================================

def _compute_spmi_row(cooc_val, cooc_col, freq_i, order_freqs,
                      total_orders, log_shift):
    """Tính dòng SPMI cho một sản phẩm.

    Duyệt qua tất cả co-occurrence của sản phẩm i, tính SPMI,
    chỉ giữ entry có SPMI > 0.

    Tham số
    ----------
    cooc_val : numpy array — giá trị co-occurrence trên dòng i
    cooc_col : numpy array — chỉ số cột
    freq_i : float — order frequency của sản phẩm i
    order_freqs : numpy array — order frequency toàn cục
    total_orders : int
    log_shift : float — log(k)

    Trả về
    -------
    tuple: (row_scores, row_cols)
        row_scores: list of float (SPMI > 0)
        row_cols: list of int (cột tương ứng)
    """
    row_scores = []
    row_cols   = []

    for j, cooc_cnt in zip(cooc_col, cooc_val):
        freq_j = order_freqs[j]
        if freq_j == 0 or cooc_cnt == 0:
            continue

        # Công thức PMI
        pmi     = np.log(cooc_cnt * total_orders / (freq_i * freq_j))
        spmi_val = max(pmi - log_shift, 0)

        if spmi_val > 0:
            row_scores.append(spmi_val)
            row_cols.append(j)

    return row_scores, row_cols


def compute_spmi(cooc_matrix, order_freqs, total_orders, k=1):
    """
    Tính SPMI từ ma trận co-occurrence.

    SPMI(A,B) = max(log(cooc[A][B] * N / (freq[A] * freq[B])) - log(k), 0)

    Trong đó N = total_prior_orders, freq = document frequency (số đơn chứa sản phẩm).

    Tham số
    ----------
    cooc_matrix : csr_matrix
        Ma trận co-occurrence (n_products x n_products).
    order_freqs : numpy array
        Số đơn hàng chứa mỗi sản phẩm.
    total_orders : int
        Tổng số prior orders (3,214,874).
    k : int
        Tham số shift. k càng cao → càng conservative (ít edges).

    Trả về
    -------
    csr_matrix
        SPMI matrix, chỉ giữ entries có SPMI > 0.
    """
    n         = cooc_matrix.shape[0]
    log_shift = np.log(k)

    print(f"  Đang tính SPMI với k={k} (shift = {log_shift:.4f})...")

    spmi_data    = []
    spmi_indices = []
    spmi_indptr  = [0]

    for i in tqdm(range(n), desc="  Tính SPMI từng dòng", unit="rows"):
        row = cooc_matrix[i]
        if row.nnz == 0:
            spmi_indptr.append(spmi_indptr[-1])
            continue

        freq_i = order_freqs[i]
        if freq_i == 0:
            spmi_indptr.append(spmi_indptr[-1])
            continue

        row_scores, row_cols = _compute_spmi_row(
            row.data, row.indices, freq_i, order_freqs,
            total_orders, log_shift
        )

        spmi_data.append(np.array(row_scores, dtype=np.float32))
        spmi_indices.append(np.array(row_cols, dtype=np.int32))
        spmi_indptr.append(spmi_indptr[-1] + len(row_cols))

    # Xây dựng CSR matrix
    spmi_data    = np.concatenate(spmi_data) if spmi_data else np.array([], dtype=np.float32)
    spmi_indices = np.concatenate(spmi_indices) if spmi_indices else np.array([], dtype=np.int32)
    spmi_indptr  = np.array(spmi_indptr, dtype=np.int32)

    spmi_csr = csr_matrix((spmi_data, spmi_indices, spmi_indptr), shape=(n, n))

    del spmi_data, spmi_indices, spmi_indptr; gc.collect()

    print(f"  SPMI matrix non-zero entries: {spmi_csr.nnz:,}")
    print(f"  Trung bình non-zero mỗi sản phẩm: {spmi_csr.nnz / n:.1f}")

    return spmi_csr


# =============================================================================
#  BƯỚC 3 — Đánh giá in-sample (train)
# =============================================================================

def evaluate_in_sample(spmi_matrix, train_gt_df, ks=(5, 10, 20)):
    """
    Đánh giá in-sample trên tập train dùng leave-one-out mỗi sản phẩm.

    Với mỗi đơn hàng trong train_gt:
      - Lấy tất cả sản phẩm trong đơn đó
      - Với mỗi sản phẩm Pi làm query:
          - Ground truth = tất cả sản phẩm khác trong cùng đơn
          - Lấy top-K recommendations từ spmi_matrix[Pi]
          - Tính hit (bất kỳ match nào cũng tính là 1)

    Chỉ đánh giá đơn có >= 2 sản phẩm.

    Tham số
    ----------
    spmi_matrix : csr_matrix
        SPMI similarity (n_products x n_products).
    train_gt_df : pd.DataFrame
        Ground truth interactions cho tập train.
    ks : tuple of int

    Trả về
    -------
    dict: {k: recall_value} cho mỗi k
    """
    print(f"\n  Đánh giá in-sample trên tập train ({len(train_gt_df):,} interactions)...")

    # Nhóm sản phẩm theo đơn hàng
    order_groups  = train_gt_df.groupby("order_id")["product_id"].apply(list)

    hits          = {k: 0.0 for k in ks}
    total_queries = 0

    for order_id, products in tqdm(
        order_groups.items(),
        desc="  Đánh giá",
        unit="orders",
        total=len(order_groups),
    ):
        n = len(products)
        if n < 2:
            continue

        products_set = set(products)

        for i in range(n):
            query        = products[i]
            ground_truth = products_set - {query}

            # Lấy recommendations từ SPMI matrix
            row = spmi_matrix[query]
            if row.nnz == 0:
                continue

            # Lấy top items theo SPMI score
            row_data    = row.data
            row_indices = row.indices
            # Sắp xếp theo score giảm dần
            sorted_idx  = np.argsort(row_data)[::-1]

            max_k       = max(ks)
            top_indices = row_indices[sorted_idx[:max_k]]

            for k in ks:
                top_k_set = set(top_indices[:k].tolist())
                if top_k_set & ground_truth:
                    hits[k] += 1

            total_queries += 1

    results = {}
    print(f"  Tổng queries đánh giá: {total_queries:,}")
    for k in ks:
        recall = hits[k] / total_queries if total_queries > 0 else 0
        results[f"recall@{k}"] = round(recall, 6)
        print(f"  Recall@{k}: {recall:.4f}")

    return results


# =============================================================================
#  BƯỚC 4 — Tuning tham số k
# =============================================================================

def tune_spmi_k(cooc_matrix, order_freqs, total_orders, train_gt_df,
                k_values=(1, 2, 3, 5, 10)):
    """
    Tune tham số shift k của SPMI trên tập train.

    Tham số
    ----------
    cooc_matrix : csr_matrix
    order_freqs : numpy array
    total_orders : int
    train_gt_df : pd.DataFrame
    k_values : tuple of int

    Trả về
    -------
    tuple: (best_k, best_spmi_matrix, all_results)
    """
    print("=" * 50)
    print("Tuning tham số k của SPMI")
    print("=" * 50)

    best_k      = None
    best_score  = -1
    best_spmi   = None
    all_results = {}

    for k in k_values:
        print(f"\n--- Đang test k = {k} ---")
        spmi    = compute_spmi(cooc_matrix, order_freqs, total_orders, k=k)
        metrics = evaluate_in_sample(spmi, train_gt_df)

        # Dùng recall@5 làm tiêu chí chọn
        score = metrics.get("recall@5", 0)
        all_results[k] = metrics

        if score > best_score:
            best_score = score
            best_k     = k
            best_spmi  = spmi
            print(f"  >>> K tốt nhất mới: k = {k} (recall@5 = {score:.4f})")

    print(f"\nk tốt nhất = {best_k} với recall@5 = {best_score:.4f}")

    return best_k, best_spmi, all_results


# =============================================================================
#  Pipeline chính
# =============================================================================

def build_spmi_model(prior_df, train_gt_df, total_prior_orders=3214874,
                     k_values=(1, 2, 3, 5, 10)):
    """
    Pipeline SPMI đầy đủ: co-occurrence → PMI → tune k → lưu.

    Tham số
    ----------
    prior_df : pd.DataFrame
        Prior order products.
    train_gt_df : pd.DataFrame
        Train ground truth để tuning.
    total_prior_orders : int
        Tổng số prior orders (mặc định: 3,214,874).
    k_values : tuple

    Trả về
    -------
    tuple: (cooc_matrix, spmi_matrix, best_k, tuning_results)
    """
    print("=" * 50)
    print("Xây dựng SPMI Model")
    print("=" * 50)

    # Bước 1: Đếm co-occurrence
    print("\n[1/3] Đếm co-occurrence từ prior...")
    n_products = max(
        prior_df["product_id"].max(),
        train_gt_df["product_id"].max(),
    ) + 1
    cooc_matrix, order_freqs = count_cooccurrence(prior_df, n_products=n_products)

    # Bước 2: Tune k trên train
    print("\n[2/3] Tuning tham số k trên tập train...")
    best_k, spmi_matrix, tuning_results = tune_spmi_k(
        cooc_matrix, order_freqs, total_prior_orders, train_gt_df, k_values
    )

    # Bước 3: SPMI cuối cùng với k tốt nhất (đã làm trong tune)
    print(f"\n[3/3] SPMI cuối cùng với k tốt nhất = {best_k}")

    del order_freqs; gc.collect()

    return cooc_matrix, spmi_matrix, best_k, tuning_results


# =============================================================================
#  Lưu model
# =============================================================================

def save_model(cooc_matrix, spmi_matrix, best_k, tuning_results):
    """
    Lưu SPMI model outputs.

    Tham số
    ----------
    cooc_matrix : csr_matrix
    spmi_matrix : csr_matrix
    best_k : int
    tuning_results : dict
    """
    print("\nĐang lưu SPMI model outputs...")

    save_npz(MODELS_DIR / "cooc_matrix.npz", cooc_matrix)
    print(f"  Đã lưu: models/cooc_matrix.npz")

    save_npz(MODELS_DIR / "spmi_matrix.npz", spmi_matrix)
    print(f"  Đã lưu: models/spmi_matrix.npz")

    output = {
        "best_k": best_k,
        "tuning_results": {
            str(k): metrics for k, metrics in tuning_results.items()
        },
    }
    with open(MODELS_DIR / "spmi_best_k.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Đã lưu: models/spmi_best_k.json")

    print("\nSPMI model hoàn tất!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_order_products, load_train_test_split

    # Tải dữ liệu
    print("Đang tải prior interactions...")
    prior_df = load_order_products("prior")

    print("Đang tải train/test split...")
    train_gt_df, _ = load_train_test_split()

    # Xây dựng model
    cooc_matrix, spmi_matrix, best_k, tuning_results = build_spmi_model(
        prior_df, train_gt_df
    )

    # Lưu
    save_model(cooc_matrix, spmi_matrix, best_k, tuning_results)