"""
Knowledge Graph (KG) model — Node2Vec embeddings.

Builds a graph with:
  - Nodes: product (49,688) + department (21)
  - Edges:
    1. (product_A) --[co_purchase]--> (product_B): only pairs with SPMI > 0,
       weight = SPMI value. Uses SPMI to filter noise and reduce edges.
    2. (product) --[belongs_to]--> (department): weight = 1.0

Learns node2vec embeddings on the graph, then computes cosine similarity
between product embeddings.

Depends on:
  - src.utils.data_loader (for prior, products)
  - src.features.build_spmi (output: models/spmi_matrix.npz)

Outputs:
  - models/kg_embeddings.npy        - Product embeddings matrix
  - models/kg_best_params.json      - Best hyperparameters from grid search
  - models/kg_similarity.npz        - Cosine similarity (product x product)
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import networkx as nx
from node2vec import Node2Vec

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_graph(spmi_matrix, products_df, prior_df):
    """
    Build NetworkX graph with product and department nodes.

    Graph structure:
      - Nodes: product_<id> (type='product'), dept_<id> (type='department')
      - Edges:
        * co_purchase: product-product, weight=SPMI value (only SPMI > 0)
        * belongs_to: product-department, weight=1.0

    Parameters
    ----------
    spmi_matrix : csr_matrix
        SPMI matrix (n_products x n_products).
    products_df : pd.DataFrame
        With columns: product_id, department_id.
    prior_df : pd.DataFrame
        Prior interactions (used to determine which products appear in prior).

    Returns
    -------
    nx.Graph
    """
    print("  Building graph...")

    n_products = spmi_matrix.shape[0]
    n_departments = products_df["department_id"].max() + 1

    graph = nx.Graph()

    # Add product nodes
    product_ids_in_prior = set(prior_df["product_id"].unique())

    for pid in range(n_products):
        product_label = f"product_{pid}"
        # Mark whether product exists in prior (0 = cold start)
        in_prior = 1 if pid in product_ids_in_prior else 0
        graph.add_node(product_label, type="product", in_prior=in_prior)

    # Add department nodes
    for did in range(n_departments):
        dept_label = f"dept_{did}"
        graph.add_node(dept_label, type="department")

    # Add co_purchase edges from SPMI matrix
    print(f"  Adding co_purchase edges from SPMI (SPMI > 0)...")
    edge_count = 0
    spmi_coo = spmi_matrix.tocoo()

    for i, j, w in tqdm(
        zip(spmi_coo.row, spmi_coo.col, spmi_coo.data),
        desc="  Adding co_purchase edges",
        unit="edges",
        total=spmi_coo.nnz,
    ):
        # Only add upper triangle to avoid duplicate edges in undirected graph
        if i < j and w > 0:
            graph.add_edge(
                f"product_{i}",
                f"product_{j}",
                weight=float(w),
                edge_type="co_purchase",
            )
            edge_count += 1

    print(f"  Added {edge_count:,} co_purchase edges")

    # Add belongs_to edges (product → department)
    print(f"  Adding belongs_to edges...")
    for _, row in products_df.iterrows():
        pid = row["product_id"]
        did = row["department_id"]
        if pid < n_products and did < n_departments:
            graph.add_edge(
                f"product_{pid}",
                f"dept_{did}",
                weight=1.0,
                edge_type="belongs_to",
            )

    print(f"  Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    return graph


def train_node2vec(graph, dimensions=128, walk_length=20, num_walks=200,
                   window=10, workers=4):
    """
    Train node2vec embeddings on the graph.

    Parameters
    ----------
    graph : nx.Graph
    dimensions : int
        Embedding dimension.
    walk_length : int
        Length of each random walk.
    num_walks : int
        Number of walks per node.
    window : int
        Context window size for Word2Vec.
    workers : int
        Number of parallel workers.

    Returns
    -------
    tuple: (model, embeddings_matrix)
        model: trained Node2Vec model
        embeddings_matrix: numpy array (n_products x dimensions)
            Only product nodes, indexed by product_id.
    """
    print(f"  Training node2vec: dim={dimensions}, walk_len={walk_length}, "
          f"num_walks={num_walks}...")

    # Initialize Node2Vec
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        quiet=True,
    )

    # Train
    model = node2vec.fit(window=window, min_count=1)

    # Extract product embeddings
    n_products = max(
        int(node.replace("product_", ""))
        for node in graph.nodes()
        if node.startswith("product_")
    ) + 1

    embeddings = np.zeros((n_products, dimensions), dtype=np.float32)

    for pid in range(n_products):
        label = f"product_{pid}"
        if label in model.wv:
            embeddings[pid] = model.wv[label].astype(np.float32)

    print(f"  Embeddings shape: {embeddings.shape}")

    return model, embeddings


def compute_kg_similarity(embeddings, top_k=100):
    """
    Compute cosine similarity between product embeddings.

    Parameters
    ----------
    embeddings : numpy array (n_products x dimensions)
    top_k : int
        Keep only top-K similar products per row.

    Returns
    -------
    csr_matrix (n_products x n_products)
    """
    print("  Computing cosine similarity...")
    n = embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero for cold-start products
    embeddings_norm = embeddings / norms

    # Compute similarity in chunks to manage memory
    sim_lil = lil_matrix((n, n), dtype=np.float32)

    chunk_size = 1000
    for start in tqdm(range(0, n, chunk_size), desc="  Computing chunks"):
        end = min(start + chunk_size, n)
        chunk = embeddings_norm[start:end]
        sim_chunk = chunk @ embeddings_norm.T  # (chunk_size x n)

        for i, row_idx in enumerate(range(start, end)):
            row = sim_chunk[i]
            # Set self-similarity to 0
            row[row_idx] = 0
            # Get top-K
            top_indices = np.argpartition(row, -top_k)[-top_k:]
            top_values = row[top_indices]
            # Sort descending
            sorted_idx = np.argsort(top_values)[::-1]
            top_indices = top_indices[sorted_idx]
            top_values = top_values[sorted_idx]
            # Keep only positive
            positive = top_values > 0
            if positive.any():
                sim_lil[row_idx, top_indices[positive]] = top_values[positive]

    sim_csr = sim_lil.tocsr()
    print(f"  KG similarity: {sim_csr.shape}, non-zero: {sim_csr.nnz:,}")

    return sim_csr


def evaluate_kg_in_sample(sim_matrix, train_gt_df, ks=(5, 10, 20)):
    """
    In-sample evaluation of KG similarity on train set.

    Same leave-one-out per product as SPMI evaluation.

    Parameters
    ----------
    sim_matrix : csr_matrix
    train_gt_df : pd.DataFrame
    ks : tuple of int

    Returns
    -------
    dict: {f"recall@{k}": value}
    """
    print(f"\n  Evaluating KG in-sample ({len(train_gt_df):,} interactions)...")

    order_groups = train_gt_df.groupby("order_id")["product_id"].apply(list)

    hits = {k: 0.0 for k in ks}
    total_queries = 0

    for products in tqdm(
        order_groups.values,
        desc="  Evaluating KG",
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

            row = sim_matrix[query]
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
    print(f"\n  Total queries: {total_queries:,}")
    for k in ks:
        recall = hits[k] / total_queries if total_queries > 0 else 0
        results[f"recall@{k}"] = round(recall, 6)
        print(f"  Recall@{k}: {recall:.4f}")

    return results


def tune_kg_params(graph, train_gt_df,
                   walk_lengths=(10, 20, 30),
                   dimensions_list=(64, 128),
                   num_walks_list=(100, 200)):
    """
    Grid search node2vec hyperparameters on train set.

    Parameters
    ----------
    graph : nx.Graph
    train_gt_df : pd.DataFrame
    walk_lengths : tuple
    dimensions_list : tuple
    num_walks_list : tuple

    Returns
    -------
    tuple: (best_params, best_embeddings, best_sim, all_results)
    """
    print("=" * 50)
    print("Tuning KG (node2vec) parameters")
    print("=" * 50)

    best_score = -1
    best_params = None
    best_embeddings = None
    best_sim = None
    all_results = []

    total_combos = len(walk_lengths) * len(dimensions_list) * len(num_walks_list)
    combo_idx = 0

    for walk_length in walk_lengths:
        for dimensions in dimensions_list:
            for num_walks in num_walks_list:
                combo_idx += 1
                print(f"\n--- [{combo_idx}/{total_combos}] walk_len={walk_length}, "
                      f"dim={dimensions}, num_walks={num_walks} ---")

                # Train
                _, embeddings = train_node2vec(
                    graph,
                    dimensions=dimensions,
                    walk_length=walk_length,
                    num_walks=num_walks,
                )

                # Compute similarity
                sim = compute_kg_similarity(embeddings, top_k=100)

                # Evaluate on train
                metrics = evaluate_kg_in_sample(sim, train_gt_df)
                score = metrics.get("recall@5", 0)

                result = {
                    "walk_length": walk_length,
                    "dimensions": dimensions,
                    "num_walks": num_walks,
                    "metrics": metrics,
                }
                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "walk_length": walk_length,
                        "dimensions": dimensions,
                        "num_walks": num_walks,
                    }
                    best_embeddings = embeddings
                    best_sim = sim
                    print(f"  >>> New best! recall@5 = {score:.4f}")

    print(f"\nBest params: {best_params}")
    print(f"Best recall@5: {best_score:.4f}")

    return best_params, best_embeddings, best_sim, all_results


def build_kg_model(spmi_matrix, products_df, prior_df, train_gt_df):
    """
    Full KG pipeline: graph → node2vec → similarity → tune → save.

    Parameters
    ----------
    spmi_matrix : csr_matrix
    products_df : pd.DataFrame
    prior_df : pd.DataFrame
    train_gt_df : pd.DataFrame

    Returns
    -------
    tuple: (best_params, embeddings, similarity, tuning_results)
    """
    print("=" * 50)
    print("Building Knowledge Graph (KG) Model")
    print("=" * 50)

    # Step 1: Build graph
    print("\n[1/3] Building graph from SPMI + product metadata...")
    graph = build_graph(spmi_matrix, products_df, prior_df)

    # Step 2: Tune node2vec on train
    print("\n[2/3] Tuning node2vec hyperparameters...")
    best_params, embeddings, sim_matrix, tuning_results = tune_kg_params(
        graph, train_gt_df
    )

    # Step 3: Done
    print(f"\n[3/3] KG model complete with best params: {best_params}")

    return best_params, embeddings, sim_matrix, tuning_results


def save_model(best_params, embeddings, sim_matrix, tuning_results):
    """
    Save KG model outputs.

    Parameters
    ----------
    best_params : dict
    embeddings : numpy array
    sim_matrix : csr_matrix
    tuning_results : list
    """
    print("\nSaving KG model outputs...")

    np.save(MODELS_DIR / "kg_embeddings.npy", embeddings)
    print(f"  Saved: models/kg_embeddings.npy")

    with open(MODELS_DIR / "kg_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved: models/kg_best_params.json")

    save_npz(MODELS_DIR / "kg_similarity.npz", sim_matrix)
    print(f"  Saved: models/kg_similarity.npz")

    # Also save tuning results for reference
    with open(MODELS_DIR / "kg_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"  Saved: models/kg_tuning_results.json")

    print("\nKG model complete!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_products, load_order_products, load_train_test_split

    # Load data
    print("Loading data...")
    products_df = load_products()
    prior_df = load_order_products("prior")
    train_gt_df, _ = load_train_test_split()

    # Load SPMI matrix from previous step
    print("Loading SPMI matrix...")
    spmi_matrix = load_npz(MODELS_DIR / "spmi_matrix.npz")

    # Build model
    best_params, embeddings, sim_matrix, tuning_results = build_kg_model(
        spmi_matrix, products_df, prior_df, train_gt_df
    )

    # Save
    save_model(best_params, embeddings, sim_matrix, tuning_results)