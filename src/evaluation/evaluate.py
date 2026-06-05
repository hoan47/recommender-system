"""
Final evaluation — compare all models on TEST set.

This is the ONLY place where test set (75,000 orders) is used.
All hyperparameter tuning was done exclusively on train set.

Evaluation protocol (Leave-One-Out per Product):
  For each order in test:
    products = [P1, P2, ..., Pn]
    If n < 2 → skip (no ground truth)
    For each Pi as query:
      ground_truth = other products in the same order
      recommendations = model.top_k(Pi)
      Compute recall@K, NDCG@K, MAP@K

Metrics reported at K = 5, 10, 20:
  - Recall@K: proportion of ground truth items found in top-K
  - NDCG@K:  Normalized Discounted Cumulative Gain
  - MAP@K:   Mean Average Precision

Output:
  - results/metrics.json — comparison table of all 4 models
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def recall_at_k(recommended, ground_truth, k):
    """
    Recall@K: proportion of ground truth items found in top-K recommendations.

    Parameters
    ----------
    recommended : list of int
        Top-K recommended product IDs (ordered by score).
    ground_truth : set of int
        Set of true product IDs in the order.
    k : int

    Returns
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
    where rel_i = 1 if item i is in ground truth, else 0.

    NDCG@K = DCG@K / IDCG@K

    Parameters
    ----------
    recommended : list of int
    ground_truth : set of int
    k : int

    Returns
    -------
    float: NDCG value ∈ [0, 1]
    """
    if len(ground_truth) == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0 for pos 1

    # IDCG (ideal: all ground truth items at top)
    idcg = 0.0
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def map_at_k(recommended, ground_truth, k):
    """
    MAP@K: Mean Average Precision at K.

    AP@K = (1 / min(K, |ground_truth|)) * Σ P(i) * rel(i)
    where P(i) = precision at position i, rel(i) = 1 if hit.

    MAP@K = mean of AP@K across all queries.

    Parameters
    ----------
    recommended : list of int
    ground_truth : set of int
    k : int

    Returns
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
    Get top-K recommended products for a query product from a model matrix.

    Parameters
    ----------
    model_matrix : csr_matrix
        Item-item score matrix (n_products x n_products).
    query_product : int
        Product ID to query.
    k : int
        Number of recommendations.

    Returns
    -------
    list of int: Top-K product IDs sorted by score descending.
    """
    row = model_matrix[query_product]
    if row.nnz == 0:
        return []

    # Sort by score descending
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
    Evaluate a single model on the full test set.

    Leave-One-Out per product protocol:
      - For each order, each product is used as a query once
      - Ground truth = all other products in the same order
      - Metrics aggregated across all queries

    Parameters
    ----------
    model_matrix : csr_matrix
        Item-item score matrix.
    test_gt_df : pd.DataFrame
        Test ground truth (order_id, product_id).
    ks : tuple of int
        K values for evaluation.
    model_name : str
        Name for progress display.

    Returns
    -------
    dict: metrics at each K value
    """
    print(f"\n{'=' * 50}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 50}")

    # Group products by order
    order_groups = test_gt_df.groupby("order_id")["product_id"].apply(list)

    # Accumulate metrics
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

            # Get top-K recommendations
            max_k = max(ks)
            recommended = get_top_k_recommendations(model_matrix, query, max_k)

            if len(recommended) == 0:
                zero_recommendation_queries += 1
                # If no recommendations, all metrics = 0
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

    # Aggregate results
    results = {}
    print(f"\n  Total queries: {total_queries:,}")
    print(f"  Skipped orders (n<2): {skipped_orders:,}")
    print(f"  Zero-recommendation queries: {zero_recommendation_queries:,} "
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
    Load and evaluate all 4 models: CB, SPMI, KG, Hybrid.

    Parameters
    ----------
    test_gt_df : pd.DataFrame
        Test ground truth.

    Returns
    -------
    dict: {model_name: metrics}
    """
    print("=" * 60)
    print("FINAL EVALUATION — TEST SET (75,000 orders)")
    print("⚠️  This is the ONLY time test data is used!")
    print("=" * 60)

    models = {}

    # Load CB
    cb_path = MODELS_DIR / "item_similarity_cb.npz"
    if cb_path.exists():
        print("\nLoading CB model...")
        models["CB"] = load_npz(cb_path)
    else:
        print("\n⚠️  CB model not found, skipping.")

    # Load SPMI
    spmi_path = MODELS_DIR / "spmi_matrix.npz"
    if spmi_path.exists():
        print("Loading SPMI model...")
        models["SPMI"] = load_npz(spmi_path)
    else:
        print("⚠️  SPMI model not found, skipping.")

    # Load KG
    kg_path = MODELS_DIR / "kg_similarity.npz"
    if kg_path.exists():
        print("Loading KG model...")
        models["KG"] = load_npz(kg_path)
    else:
        print("⚠️  KG model not found, skipping.")

    # Load Hybrid
    hybrid_path = MODELS_DIR / "hybrid_matrix.npz"
    if hybrid_path.exists():
        print("Loading Hybrid model...")
        models["Hybrid"] = load_npz(hybrid_path)
    else:
        print("⚠️  Hybrid model not found, skipping.")

    print(f"\nLoaded {len(models)} model(s). Starting evaluation...")

    # Evaluate each model
    all_results = {}

    for name, matrix in models.items():
        metrics = evaluate_model(matrix, test_gt_df, ks=(5, 10, 20), model_name=name)
        all_results[name] = metrics

    # Summary comparison table
    print("\n" + "=" * 60)
    print("SUMMARY: All Models Comparison")
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
    Save evaluation results to results/metrics.json.

    Parameters
    ----------
    all_results : dict
    """
    output_path = RESULTS_DIR / "metrics.json"

    # Format for readability
    output = {
        "description": "Final evaluation on test set (75,000 orders)",
        "protocol": "Leave-One-Out per product",
        "ks": [5, 10, 20],
        "metrics_per_model": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def generate_summary_table(all_results):
    """
    Generate a markdown-formatted summary table and save to results/summary.md.

    Parameters
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

    print(f"Summary table saved to: {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_train_test_split

    # Load test ground truth
    print("Loading test data (75,000 orders)...")
    _, test_gt_df = load_train_test_split()
    print(f"  Test ground truth: {len(test_gt_df):,} interactions")
    print(f"  Unique orders: {test_gt_df['order_id'].nunique():,}")
    print(f"  Unique products: {test_gt_df['product_id'].nunique():,}")

    # Compare all models
    all_results = compare_models(test_gt_df)

    # Save results
    save_results(all_results)
    generate_summary_table(all_results)

    print("\n✅ Evaluation complete!")