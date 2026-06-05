"""
Shifted Positive PMI (SPMI) — Collaborative Filtering model.

Builds item-item co-occurrence from prior order history, then applies
SPMI to filter noise and identify truly complementary product pairs.

Co-occurrence counting:
  - For each order in prior, every pair of products (A, B) in that order
    counts as one co-occurrence.
  - Processed chunk-by-chunk to handle 32.4M records.

SPMI formula:
  PMI(A,B) = log(cooc[A][B] * total_prior_orders / (freq[A] * freq[B]))
  SPMI(A,B) = max(PMI(A,B) - log(k), 0)

  where freq[A] = number of orders containing product A (document frequency),
  NOT total purchase count. total_prior_orders = 3,214,874.

Hyperparameter tuning:
  - k values: [1, 2, 3, 5, 10]
  - Tuned on TRAIN set only (in-sample leave-one-out)
  - TEST set is NEVER used during tuning

Depends on: src.utils.data_loader

Outputs:
  - models/cooc_matrix.npz      - Raw co-occurrence sparse matrix
  - models/spmi_matrix.npz      - SPMI sparse matrix (only SPMI > 0)
  - models/spmi_best_k.json     - Best k value and train metrics
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def count_cooccurrence(prior_df, n_products=None):
    """
    Count co-occurrence pairs from prior orders.

    For each order, every unordered pair (A, B) of products increments
    cooc[A][B] and cooc[B][A] by 1.

    Parameters
    ----------
    prior_df : pd.DataFrame
        Columns: order_id, product_id (from order_products__prior.csv).
    n_products : int or None
        Number of unique products (if None, inferred from data).

    Returns
    -------
    tuple: (cooc_matrix, order_freqs)
        cooc_matrix: scipy.sparse.csr_matrix (n_products x n_products)
            cooc[A][B] = number of orders where both A and B appear.
        order_freqs: numpy array (n_products,)
            order_freqs[A] = number of orders containing product A.
    """
    if n_products is None:
        n_products = prior_df["product_id"].max() + 1

    print(f"  Number of products (index space): {n_products:,}")

    # Use LIL matrix for efficient incremental building
    cooc = lil_matrix((n_products, n_products), dtype=np.float64)
    order_freqs = np.zeros(n_products, dtype=np.float64)

    # Process by orders
    grouped = prior_df.groupby("order_id")
    n_orders = prior_df["order_id"].nunique()

    print(f"  Processing {n_orders:,} orders ({len(prior_df):,} interactions)...")

    for order_id, group in tqdm(grouped, desc="  Counting co-occurrence", unit="orders"):
        products = group["product_id"].values
        n = len(products)

        # Skip orders with only 1 product (no pairs)
        if n < 2:
            continue

        # Update order frequencies (each product in this order → freq++)
        unique_products = np.unique(products)
        order_freqs[unique_products] += 1

        # Count all unordered pairs
        for i in range(n):
            for j in range(i + 1, n):
                a, b = int(products[i]), int(products[j])
                cooc[a, b] += 1
                cooc[b, a] += 1

    print(f"  Co-occurrence matrix built. Converting to CSR...")
    cooc_csr = cooc.tocsr()
    print(f"  Non-zero cooc entries: {cooc_csr.nnz:,}")
    print(f"  Average non-zero per product: {cooc_csr.nnz / n_products:.1f}")

    return cooc_csr, order_freqs


def compute_spmi(cooc_matrix, order_freqs, total_orders, k=1):
    """
    Compute SPMI from co-occurrence matrix.

    SPMI(A,B) = max(log(cooc[A][B] * N / (freq[A] * freq[B])) - log(k), 0)

    where N = total_prior_orders, freq = document frequency (#orders containing product).

    Parameters
    ----------
    cooc_matrix : csr_matrix
        Co-occurrence matrix (n_products x n_products).
    order_freqs : numpy array
        Number of orders containing each product.
    total_orders : int
        Total number of prior orders (3,214,874).
    k : int
        Shift parameter. Higher k → more conservative (fewer edges).

    Returns
    -------
    csr_matrix
        SPMI matrix, only entries with SPMI > 0 are kept.
    """
    print(f"  Computing SPMI with k={k} (shift = {np.log(k):.4f})...")

    n = cooc_matrix.shape[0]
    spmi = cooc_matrix.astype(np.float64).tolil()
    log_shift = np.log(k)

    # Build frequency product matrix for normalization
    # P(A) * P(B) = (freq[A]/N) * (freq[B]/N)
    # cooc[A][B] * N / (freq[A] * freq[B]) → log

    # We process row by row for memory efficiency
    spmi_data = []
    spmi_indices = []
    spmi_indptr = [0]

    for i in tqdm(range(n), desc="  Computing SPMI rows", unit="rows"):
        row = cooc_matrix[i]
        if row.nnz == 0:
            spmi_indptr.append(spmi_indptr[-1])
            continue

        cols = row.indices
        cooc_vals = row.data
        freq_i = order_freqs[i]

        if freq_i == 0:
            spmi_indptr.append(spmi_indptr[-1])
            continue

        row_scores = []
        row_cols = []

        for j, cooc_val in zip(cols, cooc_vals):
            freq_j = order_freqs[j]
            if freq_j == 0 or cooc_val == 0:
                continue

            # PMI formula
            pmi = np.log(cooc_val * total_orders / (freq_i * freq_j))
            spmi_val = max(pmi - log_shift, 0)

            if spmi_val > 0:
                row_scores.append(spmi_val)
                row_cols.append(j)

        spmi_data.append(np.array(row_scores, dtype=np.float32))
        spmi_indices.append(np.array(row_cols, dtype=np.int32))
        spmi_indptr.append(spmi_indptr[-1] + len(row_cols))

    # Build CSR matrix
    spmi_data = np.concatenate(spmi_data) if spmi_data else np.array([], dtype=np.float32)
    spmi_indices = np.concatenate(spmi_indices) if spmi_indices else np.array([], dtype=np.int32)
    spmi_indptr = np.array(spmi_indptr, dtype=np.int32)

    spmi_csr = csr_matrix((spmi_data, spmi_indices, spmi_indptr), shape=(n, n))

    print(f"  SPMI matrix non-zero entries: {spmi_csr.nnz:,}")
    print(f"  Average non-zero per product: {spmi_csr.nnz / n:.1f}")

    return spmi_csr


def evaluate_in_sample(spmi_matrix, train_gt_df, ks=(5, 10, 20)):
    """
    In-sample evaluation on train set using leave-one-out per product.

    For each order in train_gt:
      - Get all products in that order
      - For each product Pi as query:
          - Ground truth = all other products in the same order
          - Get top-K recommendations from spmi_matrix[Pi]
          - Compute hit (any match counts as 1)

    Only evaluates orders with >= 2 products.

    Parameters
    ----------
    spmi_matrix : csr_matrix
        SPMI similarity (n_products x n_products).
    train_gt_df : pd.DataFrame
        Ground truth interactions for train set.
    ks : tuple of int

    Returns
    -------
    dict: {k: recall_value} for each k
    """
    print(f"\n  Evaluating in-sample on train set ({len(train_gt_df):,} interactions)...")

    # Group products by order
    order_groups = train_gt_df.groupby("order_id")["product_id"].apply(list)

    hits = {k: 0.0 for k in ks}
    total_queries = 0

    for order_id, products in tqdm(
        order_groups.items(),
        desc="  Evaluating",
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

            # Get recommendations from SPMI matrix
            row = spmi_matrix[query]
            if row.nnz == 0:
                continue

            # Get top items by SPMI score
            row_data = row.data
            row_indices = row.indices
            # Sort by score descending
            sorted_idx = np.argsort(row_data)[::-1]

            max_k = max(ks)
            top_indices = row_indices[sorted_idx[:max_k]]

            for k in ks:
                top_k_set = set(top_indices[:k].tolist())
                if top_k_set & ground_truth:
                    hits[k] += 1

            total_queries += 1

    results = {}
    print(f"\n  Total queries evaluated: {total_queries:,}")
    for k in ks:
        recall = hits[k] / total_queries if total_queries > 0 else 0
        results[f"recall@{k}"] = round(recall, 6)
        print(f"  Recall@{k}: {recall:.4f}")

    return results


def tune_spmi_k(cooc_matrix, order_freqs, total_orders, train_gt_df,
                k_values=(1, 2, 3, 5, 10)):
    """
    Tune SPMI shift parameter k on train set.

    Parameters
    ----------
    cooc_matrix : csr_matrix
    order_freqs : numpy array
    total_orders : int
    train_gt_df : pd.DataFrame
    k_values : tuple of int

    Returns
    -------
    tuple: (best_k, best_spmi_matrix, all_results)
    """
    print("=" * 50)
    print("Tuning SPMI k parameter")
    print("=" * 50)

    best_k = None
    best_score = -1
    best_spmi = None
    all_results = {}

    for k in k_values:
        print(f"\n--- Testing k = {k} ---")
        spmi = compute_spmi(cooc_matrix, order_freqs, total_orders, k=k)
        metrics = evaluate_in_sample(spmi, train_gt_df)

        # Use recall@5 as the selection criteria
        score = metrics.get("recall@5", 0)
        all_results[k] = metrics

        if score > best_score:
            best_score = score
            best_k = k
            best_spmi = spmi
            print(f"  >>> New best k = {k} (recall@5 = {score:.4f})")

    print(f"\nBest k = {best_k} with recall@5 = {best_score:.4f}")

    return best_k, best_spmi, all_results


def build_spmi_model(prior_df, train_gt_df, total_prior_orders=3214874,
                     k_values=(1, 2, 3, 5, 10)):
    """
    Full SPMI pipeline: co-occurrence → PMI → tune k → save.

    Parameters
    ----------
    prior_df : pd.DataFrame
        Prior order products.
    train_gt_df : pd.DataFrame
        Train ground truth for tuning.
    total_prior_orders : int
        Total number of prior orders (default: 3,214,874).
    k_values : tuple

    Returns
    -------
    tuple: (cooc_matrix, spmi_matrix, best_k, tuning_results)
    """
    print("=" * 50)
    print("Building SPMI Model")
    print("=" * 50)

    # Step 1: Count co-occurrence
    print("\n[1/3] Counting co-occurrence from prior...")
    n_products = max(
        prior_df["product_id"].max(),
        train_gt_df["product_id"].max(),
    ) + 1
    cooc_matrix, order_freqs = count_cooccurrence(prior_df, n_products=n_products)

    # Step 2: Tune k on train
    print("\n[2/3] Tuning k parameter on train set...")
    best_k, spmi_matrix, tuning_results = tune_spmi_k(
        cooc_matrix, order_freqs, total_prior_orders, train_gt_df, k_values
    )

    # Step 3: Final SPMI with best k (already done in tune)
    print(f"\n[3/3] Final SPMI with best k = {best_k}")

    return cooc_matrix, spmi_matrix, best_k, tuning_results


def save_model(cooc_matrix, spmi_matrix, best_k, tuning_results):
    """
    Save SPMI model outputs.

    Parameters
    ----------
    cooc_matrix : csr_matrix
    spmi_matrix : csr_matrix
    best_k : int
    tuning_results : dict
    """
    print("\nSaving SPMI model outputs...")

    save_npz(MODELS_DIR / "cooc_matrix.npz", cooc_matrix)
    print(f"  Saved: models/cooc_matrix.npz")

    save_npz(MODELS_DIR / "spmi_matrix.npz", spmi_matrix)
    print(f"  Saved: models/spmi_matrix.npz")

    output = {
        "best_k": best_k,
        "tuning_results": {
            str(k): metrics for k, metrics in tuning_results.items()
        },
    }
    with open(MODELS_DIR / "spmi_best_k.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: models/spmi_best_k.json")

    print("\nSPMI model complete!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_order_products, load_train_test_split

    # Load data
    print("Loading prior interactions...")
    prior_df = load_order_products("prior")

    print("Loading train/test split...")
    train_gt_df, _ = load_train_test_split()

    # Build model
    cooc_matrix, spmi_matrix, best_k, tuning_results = build_spmi_model(
        prior_df, train_gt_df
    )

    # Save
    save_model(cooc_matrix, spmi_matrix, best_k, tuning_results)