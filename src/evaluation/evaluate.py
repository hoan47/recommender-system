"""
Đánh giá cuối — so sánh tất cả models trên TEST set.

Đây là nơi DUY NHẤT tập test (75,000 đơn) được dùng.
Tất cả hyperparameter tuning đã được thực hiện trên tập train.

Giao thức đánh giá (Leave-One-Out mỗi sản phẩm):
  Với mỗi đơn hàng trong test:
    products = [P1, P2, ..., Pn]
    Nếu n < 2 → bỏ qua (không có ground truth)
    Với mỗi Pi làm query:
      ground_truth = các sản phẩm khác trong cùng đơn
      recommendations = model.top_k(Pi)
      Tính recall@K, NDCG@K, MAP@K

Metrics báo cáo tại K = 5, 10, 20:
  - Recall@K: tỉ lệ ground truth items tìm thấy trong top-K
  - NDCG@K:  Normalized Discounted Cumulative Gain
  - MAP@K:   Mean Average Precision

Output:
  - results/metrics.json — bảng so sánh của tất cả 4 models
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm

# Thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def recall_at_k(recommended, ground_truth, k):
    """
    Recall@K: tỉ lệ ground truth items tìm thấy trong top-K recommendations.

    Tham số
    ----------
    recommended : list of int
        Top-K sản phẩm được gợi ý (sắp xếp theo score).
    ground_truth : set of int
        Tập sản phẩm thực tế trong đơn hàng.
    k : int

    Trả về
    -------
    float: recall value ∈ [0, 1]
    """
    if len(ground_truth) == 0:
        return 0.0
    top_k = set(recommended[:k])
    intersection = top_k & ground_truth
    return len(intersection) / len(ground_truth)


def ndcg_at_k(recommended, ground_truth, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)
    trong đó rel_i = 1 nếu item i nằm trong ground truth, else 0.

    NDCG@K = DCG@K / IDCG@K

    Tham số
    ----------
    recommended : list of int
    ground_truth : set of int
    k : int

    Trả về
    -------
    float: NDCG value ∈ [0, 1]
    """
    if len(ground_truth) == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2 vì log2(1)=0 cho pos 1

    # IDCG (ideal: tất cả ground truth items ở top)
    idcg = 0.0
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def map_at_k(recommended, ground_truth, k):
    """
    MAP@K: Mean Average Precision tại K.

    AP@K = (1 / min(K, |ground_truth|)) * Σ P(i) * rel(i)
    trong đó P(i) = precision tại vị trí i, rel(i) = 1 nếu hit.

    MAP@K = trung bình của AP@K trên tất cả queries.

    Tham số
    ----------
    recommended : list of int
    ground_truth : set of int
    k : int

    Trả về
    -------
    float: AP value ∈ [0, 1]
    """
    if len(ground_truth) == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i

    denominator = min(len(ground_truth), k)
    return sum_precisions / denominator


def get_top_k_recommendations(model_matrix, query_product, k):
    """
    Lấy top-K sản phẩm được gợi ý cho một sản phẩm query từ model matrix.

    Tham số
    ----------
    model_matrix : csr_matrix
        Item-item score matrix (n_products x n_products).
    query_product : int
        Product ID để truy vấn.
    k : int
        Số lượng recommendations.

    Trả về
    -------
    list of int: Top-K product IDs sắp xếp theo score giảm dần.
    """
    row = model_matrix[query_product]
    if row.nnz == 0:
        return []

    # Sắp xếp theo score giảm dần
    row_data = row.data
    row_indices = row.indices

    if row.nnz <= k:
        sorted_idx = np.argsort(row_data)[::-1]
    else:
        top_idx = np.argpartition(row_data, -k)[-k:]
        sorted_idx = top_idx[np.argsort(row_data[top_idx])[::-1]]

    return row_indices[sorted_idx[:k]].tolist()


def evaluate_model(model_matrix, test_gt_df, ks=(5, 10, 20), model_name=""):
    """
    Đánh giá một model trên tập test đầy đủ.

    Giao thức Leave-One-Out mỗi sản phẩm:
      - Với mỗi đơn hàng, mỗi sản phẩm được dùng làm query một lần
      - Ground truth = tất cả sản phẩm khác trong cùng đơn
      - Metrics được tổng hợp trên tất cả queries

    Tham số
    ----------
    model_matrix : csr_matrix
        Item-item score matrix.
    test_gt_df : pd.DataFrame
        Test ground truth (order_id, product_id).
    ks : tuple of int
        Giá trị K cho evaluation.
    model_name : str
        Tên model hiển thị trong progress.

    Trả về
    -------
    dict: metrics tại mỗi giá trị K
    """
    print(f"\n{'=' * 50}")
    print(f"Đánh giá: {model_name}")
    print(f"{'=' * 50}")

    # Nhóm sản phẩm theo đơn hàng
    order_groups = test_gt_df.groupby("order_id")["product_id"].apply(list)

    # Tích lũy metrics
    all_recalls = {k: [] for k in ks}
    all_ndcgs = {k: [] for k in ks}
    all_maps = {k: [] for k in ks}

    skipped_orders = 0
    total_queries = 0
    zero_recommendation_queries = 0

    for order_id, products in tqdm(
        order_groups.items(),
        desc=f"  {model_name}",
        unit="orders",
        total=len(order_groups),
    ):
        n = len(products)
        if n < 2:
            skipped_orders += 1
            continue

        products_set = set(products)

        for i in range(n):
            query = products[i]
            ground_truth = products_set - {query}

            # Lấy top-K recommendations
            max_k = max(ks)
            recommended = get_top_k_recommendations(model_matrix, query, max_k)

            if len(recommended) == 0:
                zero_recommendation_queries += 1
                # Nếu không có recommendations, tất cả metrics = 0
                for k in ks:
                    all_recalls[k].append(0.0)
                    all_ndcgs[k].append(0.0)
                    all_maps[k].append(0.0)
                total_queries += 1
                continue

            total_queries += 1

            for k in ks:
                all_recalls[k].append(recall_at_k(recommended, ground_truth, k))
                all_ndcgs[k].append(ndcg_at_k(recommended, ground_truth, k))
                all_maps[k].append(map_at_k(recommended, ground_truth, k))

    # Tổng hợp kết quả
    results = {}
    print(f"\n  Tổng queries: {total_queries:,}")
    print(f"  Đơn bị bỏ qua (n<2): {skipped_orders:,}")
    print(f"  Queries không có recommendations: {zero_recommendation_queries:,} "
          f"({100 * zero_recommendation_queries / total_queries:.2f}%)"
          if total_queries > 0 else "")

    print(f"\n  #{'K':>3}  {'Recall':>8}  {'NDCG':>8}  {'MAP':>8}")
    print(f"  {'-' * 35}")

    for k in ks:
        recall = np.mean(all_recalls[k]) if all_recalls[k] else 0.0
        ndcg = np.mean(all_ndcgs[k]) if all_ndcgs[k] else 0.0
        map_val = np.mean(all_maps[k]) if all_maps[k] else 0.0

        results[f"recall@{k}"] = round(float(recall), 6)
        results[f"ndcg@{k}"] = round(float(ndcg), 6)
        results[f"map@{k}"] = round(float(map_val), 6)

        print(f"  {k:>3}  {recall:>8.4f}  {ndcg:>8.4f}  {map_val:>8.4f}")

    return results


def compare_models(test_gt_df):
    """
    Tải và đánh giá tất cả 4 models: CB, SPMI, KG, Hybrid.

    Tham số
    ----------
    test_gt_df : pd.DataFrame
        Test ground truth.

    Trả về
    -------
    dict: {model_name: metrics}
    """
    print("=" * 60)
    print("ĐÁNH GIÁ CUỐI — TEST SET (75,000 đơn)")
    print("⚠️  Đây là lần DUY NHẤT test data được dùng!")
    print("=" * 60)

    models = {}

    # Tải CB
    cb_path = MODELS_DIR / "item_similarity_cb.npz"
    if cb_path.exists():
        print("\nĐang tải CB model...")
        models["CB"] = load_npz(cb_path)
    else:
        print("\n⚠️  Không tìm thấy CB model, bỏ qua.")

    # Tải SPMI
    spmi_path = MODELS_DIR / "spmi_matrix.npz"
    if spmi_path.exists():
        print("Đang tải SPMI model...")
        models["SPMI"] = load_npz(spmi_path)
    else:
        print("⚠️  Không tìm thấy SPMI model, bỏ qua.")

    # Tải KG
    kg_path = MODELS_DIR / "kg_similarity.npz"
    if kg_path.exists():
        print("Đang tải KG model...")
        models["KG"] = load_npz(kg_path)
    else:
        print("⚠️  Không tìm thấy KG model, bỏ qua.")

    # Tải Hybrid
    hybrid_path = MODELS_DIR / "hybrid_matrix.npz"
    if hybrid_path.exists():
        print("Đang tải Hybrid model...")
        models["Hybrid"] = load_npz(hybrid_path)
    else:
        print("⚠️  Không tìm thấy Hybrid model, bỏ qua.")

    print(f"\nĐã tải {len(models)} model(s). Bắt đầu đánh giá...")

    # Đánh giá từng model
    all_results = {}

    for name, matrix in models.items():
        metrics = evaluate_model(matrix, test_gt_df, ks=(5, 10, 20), model_name=name)
        all_results[name] = metrics

    # Bảng so sánh tổng hợp
    print("\n" + "=" * 60)
    print("TỔNG KẾT: So sánh tất cả Models")
    print("=" * 60)

    ks = [5, 10, 20]
    metrics_names = ["recall", "ndcg", "map"]

    for metric_name in metrics_names:
        print(f"\n  {metric_name.upper()}:")
        header = f"  {'Model':>8}"
        for k in ks:
            header += f"  @{k:>2}"
        print(header)
        print(f"  {'-' * (8 + 7 * len(ks))}")

        for model_name in all_results:
            row = f"  {model_name:>8}"
            for k in ks:
                key = f"{metric_name}@{k}"
                val = all_results[model_name].get(key, 0)
                row += f"  {val:.4f}"
            print(row)

    return all_results


def save_results(all_results):
    """
    Lưu kết quả đánh giá vào results/metrics.json.

    Tham số
    ----------
    all_results : dict
    """
    output_path = RESULTS_DIR / "metrics.json"

    # Format cho dễ đọc
    output = {
        "description": "Đánh giá cuối trên test set (75,000 đơn)",
        "protocol": "Leave-One-Out mỗi sản phẩm",
        "ks": [5, 10, 20],
        "metrics_per_model": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nKết quả đã lưu tại: {output_path}")


def generate_summary_table(all_results):
    """
    Tạo bảng tổng kết dạng markdown và lưu vào results/summary.md.

    Tham số
    ----------
    all_results : dict
    """
    output_path = RESULTS_DIR / "summary.md"

    lines = []
    lines.append("# 📊 So sánh tất cả models — Test Set (75,000 đơn)")
    lines.append("")
    lines.append("## Recall@K")
    lines.append("")
    lines.append("| Model | @5 | @10 | @20 |")
    lines.append("|-------|-----|------|------|")

    for name in all_results:
        r5 = all_results[name].get("recall@5", 0)
        r10 = all_results[name].get("recall@10", 0)
        r20 = all_results[name].get("recall@20", 0)
        lines.append(f"| {name} | {r5:.4f} | {r10:.4f} | {r20:.4f} |")

    lines.append("")
    lines.append("## NDCG@K")
    lines.append("")
    lines.append("| Model | @5 | @10 | @20 |")
    lines.append("|-------|-----|------|------|")

    for name in all_results:
        n5 = all_results[name].get("ndcg@5", 0)
        n10 = all_results[name].get("ndcg@10", 0)
        n20 = all_results[name].get("ndcg@20", 0)
        lines.append(f"| {name} | {n5:.4f} | {n10:.4f} | {n20:.4f} |")

    lines.append("")
    lines.append("## MAP@K")
    lines.append("")
    lines.append("| Model | @5 | @10 | @20 |")
    lines.append("|-------|-----|------|------|")

    for name in all_results:
        m5 = all_results[name].get("map@5", 0)
        m10 = all_results[name].get("map@10", 0)
        m20 = all_results[name].get("map@20", 0)
        lines.append(f"| {name} | {m5:.4f} | {m10:.4f} | {m20:.4f} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Bảng tổng kết đã lưu tại: {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_train_test_split

    # Tải test ground truth
    print("Đang tải test data (75,000 đơn)...")
    _, test_gt_df = load_train_test_split()
    print(f"  Test ground truth: {len(test_gt_df):,} interactions")
    print(f"  Đơn hàng unique: {test_gt_df['order_id'].nunique():,}")
    print(f"  Sản phẩm unique: {test_gt_df['product_id'].nunique():,}")

    # So sánh tất cả models
    all_results = compare_models(test_gt_df)

    # Lưu kết quả
    save_results(all_results)
    generate_summary_table(all_results)

    print("\n✅ Đánh giá hoàn tất!")