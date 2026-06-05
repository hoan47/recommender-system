"""
Hybrid model: combines SPMI + KG, filtered by CB.

Combines complementary (SPMI) and knowledge-based (KG) signals,
with CB as a substitute filter.

Formula:
  final_score(A → B) = α * spmi_norm(A,B) + β * kg_sim(A,B)
  If cb_sim(A,B) > cb_threshold → final_score = 0 (remove substitute)

Hyperparameter grid:
  α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  β  ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  cb_threshold ∈ {0.7, 0.8, 0.9}

Tuned on TRAIN set only. TEST is never used.

Depends on:
  - src.features.build_tfidf (output: models/item_similarity_cb.npz)
  - src.features.build_spmi (output: models/spmi_matrix.npz)
  - src.features.build_knowledge_graph (output: models/kg_similarity.npz)

Outputs:
  - models/hybrid_best_params.json  - Best {α, β, cb_threshold, metrics}
  - models/hybrid_matrix.npz        - (Optional) Final hybrid score matrix
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_sparse(matrix):
    """
    Normalize sparse matrix values to [0, 1] using max value.

    SPMI values can be > 1, while cosine similarities (CB, KG) are in [0, 1].
    Normalizing SPMI ensures comparable scales for weighted combination.

    Parameters
    ----------
    matrix : csr_matrix

    Returns
    -------
    csr_matrix with values scaled to [0, 1]
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
    Compute hybrid recommendation score matrix.

    final_score(A → B) = alpha * spmi_norm(A,B) + beta * kg_sim(A,B)
    Set to 0 if cb_sim(A,B) > cb_threshold (substitute removal).

    Parameters
    ----------
    spmi_norm : csr_matrix
        Normalized SPMI matrix.
    kg_sim : csr_matrix
        KG similarity matrix.
    cb_sim : csr_matrix
        CB similarity matrix (used as filter).
    alpha : float
        Weight for SPMI component.
    beta : float
        Weight for KG component.
    cb_threshold : float
        CB similarity threshold for substitute removal.

    Returns
    -------
    csr_matrix
        Hybrid score matrix.
    """
    print(f"  Computing hybrid: α={alpha}, β={beta}, cb_threshold={cb_threshold}")

    n = spmi_norm.shape[0]
    hybrid = lil_matrix((n, n), dtype=np.float32)

    for i in tqdm(range(n), desc="  Building hybrid matrix", unit="rows"):
        # Get scores from each source
        spmi_row = spmi_norm[i].tocoo() if spmi_norm[i].nnz > 0 else None
        kg_row = kg_sim[i].tocoo() if kg_sim[i].nnz > 0 else None
        cb_row = cb_sim[i].tocoo() if cb_sim[i].nnz > 0 else None

        # Collect candidate indices
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

        # Apply CB filter
        cb_dict = {}
        if cb_row is not None:
            for j, val in zip(cb_row.col, cb_row.data):
                cb_dict[j] = val

        # Build row scores
        row_data = []
        row_cols = []
        for j, score in candidates.items():
            # Check CB filter
            if j in cb_dict and cb_dict[j] > cb_threshold:
                continue  # This is a substitute, remove
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
    In-sample evaluation of hybrid on train set.

    Same leave-one-out per product evaluation protocol.

    Parameters
    ----------
    hybrid_matrix : csr_matrix
    train_gt_df : pd.DataFrame
    ks : tuple of int

    Returns
    -------
    dict: {f"recall@{k}": value}
    """
    print(f"  Evaluating hybrid in-sample ({len(train_gt_df):,} interactions)...")

    order_groups = train_gt_df.groupby("order_id")["product_id"].apply(list)

    hits = {k: 0.0 for k in ks}
    total_queries = 0

    for products in tqdm(
        order_groups.values,
        desc="  Evaluating hybrid",
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
    print(f"  Total queries: {total_queries:,}")
    for k in ks:
        recall = hits[k] / total_queries if total_queries > 0 else 0
        results[f"recall@{k}"] = round(recall, 6)
        print(f"  Recall@{k}: {recall:.4f}")

    return results


def grid_search_hybrid(spmi_norm, kg_sim, cb_sim, train_gt_df):
    """
    Grid search for best hybrid weights and CB threshold.

    Searches:
      α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
      β  ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
      cb_threshold ∈ {0.7, 0.8, 0.9}

    Selection criteria: best recall@5 on train.

    Parameters
    ----------
    spmi_norm : csr_matrix
    kg_sim : csr_matrix
    cb_sim : csr_matrix
    train_gt_df : pd.DataFrame

    Returns
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

                # Skip combination that evaluates to zero
                if alpha == 0 and beta == 0:
                    print("  Skipping (α=0, β=0)")
                    continue

                # Compute hybrid matrix
                hybrid = compute_hybrid_score(
                    spmi_norm, kg_sim, cb_sim, alpha, beta, cb_thresh
                )

                # Evaluate
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
                    print(f"  >>> New best! recall@5 = {score:.4f}")

    print(f"\nBest params: {best_params}")
    print(f"Best recall@5: {best_score:.4f}")

    return best_params, best_matrix, all_results


def build_hybrid_model(spmi_matrix, kg_sim, cb_sim, train_gt_df):
    """
    Full hybrid pipeline: normalize → grid search → save.

    Parameters
    ----------
    spmi_matrix : csr_matrix
    kg_sim : csr_matrix
    cb_sim : csr_matrix
    train_gt_df : pd.DataFrame

    Returns
    -------
    tuple: (best_params, hybrid_matrix, grid_results)
    """
    print("=" * 50)
    print("Building Hybrid Model")
    print("=" * 50)

    # Step 1: Normalize SPMI to [0, 1]
    print("\n[1/3] Normalizing SPMI to [0, 1]...")
    spmi_norm = normalize_sparse(spmi_matrix)
    print(f"  SPMI max after normalization: {spmi_norm.data.max():.4f}")

    # Ensure all matrices have same shape
    n = spmi_norm.shape[0]
    if kg_sim.shape[0] < n:
        print(f"  Padding KG from {kg_sim.shape[0]} to {n}...")
        kg_sim = kg_sim.tolil()
        kg_sim.resize(n, n)
        kg_sim = kg_sim.tocsr()
    if cb_sim.shape[0] < n:
        print(f"  Padding CB from {cb_sim.shape[0]} to {n}...")
        cb_sim = cb_sim.tolil()
        cb_sim.resize(n, n)
        cb_sim = cb_sim.tocsr()

    # Step 2: Grid search on train
    print("\n[2/3] Grid search for best α, β, cb_threshold...")
    best_params, hybrid_matrix, grid_results = grid_search_hybrid(
        spmi_norm, kg_sim, cb_sim, train_gt_df
    )

    # Step 3: Done
    print(f"\n[3/3] Hybrid complete with best params: {best_params}")

    return best_params, hybrid_matrix, grid_results


def save_model(best_params, hybrid_matrix, grid_results):
    """
    Save hybrid model outputs.

    Parameters
    ----------
    best_params : dict
    hybrid_matrix : csr_matrix
    grid_results : list
    """
    print("\nSaving Hybrid model outputs...")

    output = {
        "best_params": best_params,
        "best_recall_at_5": grid_results[-1]["metrics"].get("recall@5", 0),
    }

    with open(MODELS_DIR / "hybrid_best_params.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: models/hybrid_best_params.json")

    # Save full grid results
    with open(MODELS_DIR / "hybrid_grid_results.json", "w", encoding="utf-8") as f:
        json.dump(grid_results, f, indent=2)
    print(f"  Saved: models/hybrid_grid_results.json")

    # Save best hybrid matrix
    save_npz(MODELS_DIR / "hybrid_matrix.npz", hybrid_matrix)
    print(f"  Saved: models/hybrid_matrix.npz")

    print("\nHybrid model complete!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_train_test_split

    # Load train ground truth
    print("Loading train data...")
    train_gt_df, _ = load_train_test_split()

    # Load model matrices
    print("Loading model matrices...")
    spmi_matrix = load_npz(MODELS_DIR / "spmi_matrix.npz")
    kg_sim = load_npz(MODELS_DIR / "kg_similarity.npz")
    cb_sim = load_npz(MODELS_DIR / "item_similarity_cb.npz")

    print(f"  SPMI: {spmi_matrix.shape}, nnz={spmi_matrix.nnz:,}")
    print(f"  KG:   {kg_sim.shape}, nnz={kg_sim.nnz:,}")
    print(f"  CB:   {cb_sim.shape}, nnz={cb_sim.nnz:,}")

    # Build model
    best_params, hybrid_matrix, grid_results = build_hybrid_model(
        spmi_matrix, kg_sim, cb_sim, train_gt_df
    )

    # Save
    save_model(best_params, hybrid_matrix, grid_results)