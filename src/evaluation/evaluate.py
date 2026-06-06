"""
Đánh giá cuối — so sánh tất cả models trên TEST set.

⚠️ Đây là nơi DUY NHẤT test data được dùng.

Giao thức Leave-One-Out mỗi sản phẩm:
  Với mỗi đơn test (≥ 2 sản phẩm):
    Với mỗi sản phẩm Pi làm query:
      ground_truth = các sản phẩm còn lại trong đơn
      recommendations = top-K từ model matrix
      → tính Recall@K, NDCG@K, MAP@K

Output:
  results/metrics.json
  results/summary.md
"""

import gc
import json

import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm

from src.config import (
    MODELS_DIR, RESULTS_DIR, EVAL_KS,
    CB_SIMILARITY_FILE, SPMI_MATRIX_FILE, KG_SIMILARITY_FILE, HYBRID_MATRIX_FILE,
    METRICS_FILE, SUMMARY_FILE,
)


# ============================================================================
# Metrics
# ============================================================================

def recall_at_k(top_list, gt_set, k):
    return len(set(top_list[:k]) & gt_set) / len(gt_set) if gt_set else 0.0


def ndcg_at_k(top_list, gt_set, k):
    if not gt_set:
        return 0.0
    dcg  = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_list[:k]) if item in gt_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(top_list, gt_set, k):
    if not gt_set:
        return 0.0
    hits, total = 0, 0.0
    for i, item in enumerate(top_list[:k]):
        if item in gt_set:
            hits  += 1
            total += hits / (i + 1)
    return total / min(len(gt_set), k)


# ============================================================================
# Evaluate một model
# ============================================================================

def get_top_k(matrix, query, k):
    """Lấy top-K product IDs từ hàng query của matrix."""
    row = matrix[query]
    if row.nnz == 0:
        return []
    sorted_idx = np.argsort(row.data)[::-1]
    return row.indices[sorted_idx[:k]].tolist()


def evaluate_model(matrix, test_gt_df, model_name=""):
    """Tính Recall, NDCG, MAP tại K = 5, 10, 20 trên toàn bộ test set."""
    print(f"\n--- Đánh giá: {model_name} ---")

    order_groups = test_gt_df.groupby("order_id")["product_id"].apply(list)
    ks = EVAL_KS

    sum_recall = {k: 0.0 for k in ks}
    sum_ndcg   = {k: 0.0 for k in ks}
    sum_map    = {k: 0.0 for k in ks}
    total      = 0

    for products in tqdm(order_groups, desc=model_name, unit="đơn"):
        if len(products) < 2:
            continue
        gt_set = set(products)
        for query in products:
            gt = gt_set - {query}
            top = get_top_k(matrix, query, max(ks))
            for k in ks:
                sum_recall[k] += recall_at_k(top, gt, k)
                sum_ndcg[k]   += ndcg_at_k(top, gt, k)
                sum_map[k]    += map_at_k(top, gt, k)
            total += 1

    results = {}
    print(f"  Tổng queries: {total:,}")
    print(f"  {'K':>3}  {'Recall':>8}  {'NDCG':>8}  {'MAP':>8}")
    for k in ks:
        r = sum_recall[k] / total if total else 0
        n = sum_ndcg[k]   / total if total else 0
        m = sum_map[k]    / total if total else 0
        results[f"recall@{k}"] = round(r, 6)
        results[f"ndcg@{k}"]   = round(n, 6)
        results[f"map@{k}"]    = round(m, 6)
        print(f"  {k:>3}  {r:>8.4f}  {n:>8.4f}  {m:>8.4f}")

    return results


# ============================================================================
# Load models + compare
# ============================================================================

def compare_models(test_gt_df):
    """Tải và đánh giá tất cả 4 models."""
    print("=" * 50)
    print("ĐÁNH GIÁ CUỐI — TEST SET")
    print("=" * 50)

    model_files = {
        "CB":     CB_SIMILARITY_FILE,
        "SPMI":   SPMI_MATRIX_FILE,
        "KG":     KG_SIMILARITY_FILE,
        "Hybrid": HYBRID_MATRIX_FILE,
    }

    all_results = {}
    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"⚠️ Không tìm thấy {filename}, bỏ qua.")
            continue
        matrix = load_npz(path)
        all_results[name] = evaluate_model(matrix, test_gt_df, name)

    return all_results


# ============================================================================
# Lưu kết quả
# ============================================================================

def save_results(all_results):
    with open(RESULTS_DIR / METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nKết quả: {RESULTS_DIR / METRICS_FILE}")


def save_summary(all_results):
    ks = list(EVAL_KS)
    lines = ["# So sánh models — Test Set", ""]

    for metric in ["recall", "ndcg", "map"]:
        lines += [f"## {metric.upper()}@K", "",
                  "| Model | " + " | ".join(f"@{k}" for k in ks) + " |",
                  "|-------|" + "|".join(["------"] * len(ks)) + "|"]
        for name, res in all_results.items():
            vals = " | ".join(f"{res.get(f'{metric}@{k}', 0):.4f}" for k in ks)
            lines.append(f"| {name} | {vals} |")
        lines.append("")

    with open(RESULTS_DIR / SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Bảng tổng kết: {RESULTS_DIR / SUMMARY_FILE}")


if __name__ == "__main__":
    from src.utils.data_loader import load_train_test_split

    _, test_gt_df = load_train_test_split()
    print(f"Test GT: {len(test_gt_df):,} interactions, "
          f"{test_gt_df['order_id'].nunique():,} đơn")

    all_results = compare_models(test_gt_df)
    save_results(all_results)
    save_summary(all_results)
    print("\n✅ Đánh giá hoàn tất!")