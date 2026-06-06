"""
Hybrid model: kết hợp SPMI + KG, lọc bởi CB.

Kết hợp tín hiệu complementary (SPMI) và knowledge-based (KG),
với CB làm bộ lọc substitute.

Công thức:
  final_score(A → B) = α * spmi_norm(A,B) + β * kg_sim(A,B)
  Nếu cb_sim(A,B) > cb_threshold → final_score = 0 (loại substitute)

Grid hyperparameter:
  α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  β  ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  cb_threshold ∈ {0.7, 0.8, 0.9}

Tune trên tập TRAIN. TEST không bao giờ được dùng.

Phụ thuộc:
  - src.features.build_tfidf (output: models/item_similarity_cb.npz)
  - src.features.build_spmi (output: models/spmi_matrix.npz)
  - src.features.build_knowledge_graph (output: models/kg_similarity.npz)

Outputs:
  - models/hybrid_best_params.json  - Best {α, β, cb_threshold, metrics}
  - models/hybrid_matrix.npz        - (Tuỳ chọn) Ma trận hybrid score cuối cùng
"""

import gc
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from tqdm import tqdm

# Thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_sparse(matrix):
    """
    Chuẩn hóa giá trị sparse matrix về [0, 1] dùng max value.

    SPMI values có thể > 1, trong khi cosine similarities (CB, KG) nằm trong [0, 1].
    Chuẩn hóa SPMI đảm bảo scale tương đương cho weighted combination.

    Tham số
    ----------
    matrix : csr_matrix

    Trả về
    -------
    csr_matrix với giá trị được scale về [0, 1]
    """
    if matrix.nnz == 0:
        return matrix.copy()

    max_val = matrix.data.max()
    if max_val == 0:
        return matrix.copy()

    normalized = matrix.copy()
    normalized.data = normalized.data / max_val
    return normalized


def compute_hybrid_score(spmi_norm, kg_sim, cb_sim, alpha, beta, cb_threshold):
    """
    Tính hybrid recommendation score matrix.

    final_score(A → B) = alpha * spmi_norm(A,B) + beta * kg_sim(A,B)
    Đặt về 0 nếu cb_sim(A,B) > cb_threshold (loại substitute).

    Tham số
    ----------
    spmi_norm : csr_matrix
        SPMI matrix đã chuẩn hóa.
    kg_sim : csr_matrix
        KG similarity matrix.
    cb_sim : csr_matrix
        CB similarity matrix (dùng làm bộ lọc).
    alpha : float
        Trọng số cho SPMI.
    beta : float
        Trọng số cho KG.
    cb_threshold : float
        Ngưỡng CB similarity để loại substitute.

    Trả về
    -------
    csr_matrix
        Hybrid score matrix.
    """
    print(f"  Đang tính hybrid: α={alpha}, β={beta}, cb_threshold={cb_threshold}")

    n = spmi_norm.shape[0]
    hybrid = lil_matrix((n, n), dtype=np.float32)

    for i in tqdm(range(n), desc="  Xây dựng hybrid matrix", unit="rows"):
        # Lấy scores từ mỗi nguồn
        spmi_row = spmi_norm[i].tocoo() if spmi_norm[i].nnz > 0 else None
        kg_row = kg_sim[i].tocoo() if kg_sim[i].nnz > 0 else None
        cb_row = cb_sim[i].tocoo() if cb_sim[i].nnz > 0 else None

        # Gom candidate indices
        candidates = {}
        if spmi_row is not None:
            for j, val in zip(spmi_row.col, spmi_row.data):
                candidates[j] = alpha * val
        if kg_row is not None:
            for j, val in zip(kg_row.col, kg_row.data):
                if j in candidates:
                    candidates[j] += beta * val
                else:
                    candidates[j] = beta * val

        # Áp dụng bộ lọc CB
        cb_dict = {}
        if cb_row is not None:
            for j, val in zip(cb_row.col, cb_row.data):
                cb_dict[j] = val

        # Xây dựng row scores
        row_data = []
        row_cols = []
        for j, score in candidates.items():
            # Kiểm tra bộ lọc CB
            if j in cb_dict and cb_dict[j] > cb_threshold:
                continue  # Đây là substitute, loại bỏ
            if score > 0:
                row_data.append(score)
                row_cols.append(j)

        if row_data:
            hybrid[i, row_cols] = row_data

    hybrid_csr = hybrid.tocsr()
    print(f"  Hybrid matrix: {hybrid_csr.shape}, non-zero: {hybrid_csr.nnz:,}")

    return hybrid_csr


def evaluate_hybrid_in_sample(hybrid_matrix, train_gt_df, ks=(5, 10, 20)):
    """
    Đánh giá in-sample hybrid trên tập train.

    Cùng giao thức leave-one-out mỗi sản phẩm.

    Tham số
    ----------
    hybrid_matrix : csr_matrix
    train_gt_df : pd.DataFrame
    ks : tuple of int

    Trả về
    -------
    dict: {f"recall@{k}": value}
    """
    print(f"  Đánh giá hybrid in-sample ({len(train_gt_df):,} interactions)...")

    order_groups = train_gt_df.groupby("order_id")["product_id"].apply(list)

    hits = {k: 0.0 for k in ks}
    total_queries = 0

    for products in tqdm(
        order_groups.values,
        desc="  Đánh giá hybrid",
        unit="orders",
        total=len(order_groups),
    ):
        n = len(products)
        if n < 2:
            continue

        products_set = set(products)

        for i in range(n):
            query = products[i]
            ground_truth = products_set - {query}

            row = hybrid_matrix[query]
            if row.nnz == 0:
                continue

            row_data = row.data
            row_indices = row.indices
            sorted_idx = np.argsort(row_data)[::-1]

            max_k = max(ks)
            top_indices = row_indices[sorted_idx[:max_k]]

            for k in ks:
                top_k_set = set(top_indices[:k].tolist())
                if top_k_set & ground_truth:
                    hits[k] += 1

            total_queries += 1

    results = {}
    print(f"  Tổng queries: {total_queries:,}")
    for k in ks:
        recall = hits[k] / total_queries if total_queries > 0 else 0
        results[f"recall@{k}"] = round(recall, 6)
        print(f"  Recall@{k}: {recall:.4f}")

    return results


def grid_search_hybrid(spmi_norm, kg_sim, cb_sim, train_gt_df):
    """
    Grid search tìm trọng số hybrid và ngưỡng CB tốt nhất.

    Tìm kiếm:
      α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
      β  ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
      cb_threshold ∈ {0.7, 0.8, 0.9}

    Tiêu chí chọn: recall@5 tốt nhất trên train.

    Tham số
    ----------
    spmi_norm : csr_matrix
    kg_sim : csr_matrix
    cb_sim : csr_matrix
    train_gt_df : pd.DataFrame

    Trả về
    -------
    tuple: (best_params, best_hybrid_matrix, all_results)
    """
    print("=" * 50)
    print("Grid Search: Hybrid Parameters")
    print("=" * 50)

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cb_thresholds = [0.7, 0.8, 0.9]

    best_score = -1
    best_params = None
    best_matrix = None
    all_results = []

    total = len(alphas) * len(betas) * len(cb_thresholds)
    idx = 0

    for alpha in alphas:
        for beta in betas:
            for cb_thresh in cb_thresholds:
                idx += 1
                print(f"\n--- [{idx}/{total}] α={alpha}, β={beta}, "
                      f"cb_threshold={cb_thresh} ---")

                # Bỏ qua tổ hợp cho kết quả = 0
                if alpha == 0 and beta == 0:
                    print("  Bỏ qua (α=0, β=0)")
                    continue

                # Tính hybrid matrix
                hybrid = compute_hybrid_score(
                    spmi_norm, kg_sim, cb_sim, alpha, beta, cb_thresh
                )

                # Đánh giá
                metrics = evaluate_hybrid_in_sample(hybrid, train_gt_df)
                score = metrics.get("recall@5", 0)

                result = {
                    "alpha": alpha,
                    "beta": beta,
                    "cb_threshold": cb_thresh,
                    "metrics": metrics,
                }
                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "alpha": alpha,
                        "beta": beta,
                        "cb_threshold": cb_thresh,
                    }
                    best_matrix = hybrid
                    print(f"  >>> Kết quả tốt nhất mới! recall@5 = {score:.4f}")

    print(f"\nTham số tốt nhất: {best_params}")
    print(f"Recall@5 tốt nhất: {best_score:.4f}")

    return best_params, best_matrix, all_results


def build_hybrid_model(spmi_matrix, kg_sim, cb_sim, train_gt_df):
    """
    Pipeline hybrid đầy đủ: chuẩn hóa → grid search → lưu.

    Tham số
    ----------
    spmi_matrix : csr_matrix
    kg_sim : csr_matrix
    cb_sim : csr_matrix
    train_gt_df : pd.DataFrame

    Trả về
    -------
    tuple: (best_params, hybrid_matrix, grid_results)
    """
    print("=" * 50)
    print("Xây dựng Hybrid Model")
    print("=" * 50)

    # Bước 1: Chuẩn hóa SPMI về [0, 1]
    print("\n[1/3] Đang chuẩn hóa SPMI về [0, 1]...")
    spmi_norm = normalize_sparse(spmi_matrix)
    print(f"  SPMI max sau chuẩn hóa: {spmi_norm.data.max():.4f}")

    # Đảm bảo tất cả matrices cùng shape
    n = spmi_norm.shape[0]
    if kg_sim.shape[0] < n:
        print(f"  Đang pad KG từ {kg_sim.shape[0]} lên {n}...")
        kg_sim = kg_sim.tolil()
        kg_sim.resize(n, n)
        kg_sim = kg_sim.tocsr()
    if cb_sim.shape[0] < n:
        print(f"  Đang pad CB từ {cb_sim.shape[0]} lên {n}...")
        cb_sim = cb_sim.tolil()
        cb_sim.resize(n, n)
        cb_sim = cb_sim.tocsr()

    # Bước 2: Grid search trên train
    print("\n[2/3] Grid search tìm α, β, cb_threshold tốt nhất...")
    best_params, hybrid_matrix, grid_results = grid_search_hybrid(
        spmi_norm, kg_sim, cb_sim, train_gt_df
    )

    # Bước 3: Hoàn tất
    print(f"\n[3/3] Hybrid hoàn tất với tham số tốt nhất: {best_params}")

    return best_params, hybrid_matrix, grid_results


def save_model(best_params, hybrid_matrix, grid_results):
    """
    Lưu hybrid model outputs.

    Tham số
    ----------
    best_params : dict
    hybrid_matrix : csr_matrix
    grid_results : list
    """
    print("\nĐang lưu Hybrid model outputs...")

    output = {
        "best_params": best_params,
        "best_recall_at_5": grid_results[-1]["metrics"].get("recall@5", 0),
    }

    with open(MODELS_DIR / "hybrid_best_params.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Đã lưu: models/hybrid_best_params.json")

    # Lưu toàn bộ grid results
    with open(MODELS_DIR / "hybrid_grid_results.json", "w", encoding="utf-8") as f:
        json.dump(grid_results, f, indent=2)
    print(f"  Đã lưu: models/hybrid_grid_results.json")

    # Lưu best hybrid matrix
    save_npz(MODELS_DIR / "hybrid_matrix.npz", hybrid_matrix)
    print(f"  Đã lưu: models/hybrid_matrix.npz")

    print("\nHybrid model hoàn tất!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_train_test_split

    # Tải train ground truth
    print("Đang tải train data...")
    train_gt_df, _ = load_train_test_split()

    # Tải model matrices
    print("Đang tải model matrices...")
    spmi_matrix = load_npz(MODELS_DIR / "spmi_matrix.npz")
    kg_sim = load_npz(MODELS_DIR / "kg_similarity.npz")
    cb_sim = load_npz(MODELS_DIR / "item_similarity_cb.npz")

    print(f"  SPMI: {spmi_matrix.shape}, nnz={spmi_matrix.nnz:,}")
    print(f"  KG:   {kg_sim.shape}, nnz={kg_sim.nnz:,}")
    print(f"  CB:   {cb_sim.shape}, nnz={cb_sim.nnz:,}")

    # Xây dựng model
    best_params, hybrid_matrix, grid_results = build_hybrid_model(
        spmi_matrix, kg_sim, cb_sim, train_gt_df
    )

    # Lưu
    save_model(best_params, hybrid_matrix, grid_results)