"""
Content-Based (CB) model: TF-IDF + Cosine Similarity.

Builds item-item similarity matrix based on product name and department text.
Used as:
  - Baseline model for comparison
  - Filter to remove substitute products from SPMI/KG recommendations
  - Fallback for long-tail products with insufficient co-occurrence data

Depends on: src.utils.data_loader

Outputs:
  - models/tfidf_matrix.npz        - Raw TF-IDF sparse matrix
  - models/item_similarity_cb.npz  - Cosine similarity sparse matrix
  - models/tfidf_vectorizer.pkl    - Fitted TfidfVectorizer
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Make models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_documents(products_df):
    """
    Create text documents for TF-IDF from product metadata.

    Each document = product_name + " " + department.
    Handles missing/NaN values gracefully.

    Parameters
    ----------
    products_df : pd.DataFrame
        With columns: product_id, product_name, department

    Returns
    -------
    tuple: (documents, product_ids)
        documents: list of strings
        product_ids: numpy array aligned with documents
    """
    documents = []
    product_ids = []

    for _, row in products_df.iterrows():
        product_name = row.get("product_name", "")
        department = row.get("department", "")

        # Handle missing / NaN values
        if pd.isna(product_name) or str(product_name).strip() == "":
            product_name = "unknown product"
        if pd.isna(department) or str(department).strip() == "":
            department = "unknown department"

        # Build document
        doc = f"{product_name} {department}"
        documents.append(doc)
        product_ids.append(row["product_id"])

    return documents, np.array(product_ids)


def build_tfidf(documents, max_features=10000):
    """
    Build TF-IDF matrix from text documents.

    Parameters
    ----------
    documents : list of str
        Text documents for each product.
    max_features : int
        Maximum number of TF-IDF features (vocabulary size).

    Returns
    -------
    tuple: (tfidf_matrix, vectorizer)
        tfidf_matrix: scipy.sparse.csr_matrix (n_products x max_features)
        vectorizer: fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,  # 1 + log(tf)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    return tfidf_matrix, vectorizer


def build_similarity(tfidf_matrix, top_k=None, threshold=None):
    """
    Compute cosine similarity and optionally sparsify.

    Parameters
    ----------
    tfidf_matrix : scipy.sparse.csr_matrix
        TF-IDF vectors (n_products x n_features).
    top_k : int or None
        If set, keep only top-K most similar items for each product.
    threshold : float or None
        If set, keep only similarities above this threshold.

    Returns
    -------
    scipy.sparse.csr_matrix
        Cosine similarity (n_products x n_products), sparse format.
    """
    print("  Computing cosine similarity...")
    # cosine_similarity on sparse input returns dense array
    # We compute row by row and keep sparse for memory efficiency
    n = tfidf_matrix.shape[0]
    sim_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

    # Ensure it's csr format
    if not isinstance(sim_matrix, csr_matrix):
        sim_matrix = sim_matrix.tocsr()

    # Set diagonal to 0 (a product is trivially similar to itself)
    sim_matrix.setdiag(0)
    sim_matrix.eliminate_zeros()

    # Apply threshold if specified
    if threshold is not None:
        print(f"  Applying threshold > {threshold}...")
        sim_matrix.data[sim_matrix.data <= threshold] = 0
        sim_matrix.eliminate_zeros()

    # Keep top-K per row if specified
    if top_k is not None:
        print(f"  Keeping top-{top_k} per product...")
        sim_matrix = _keep_topk_per_row(sim_matrix, top_k)

    print(f"  Similarity matrix shape: {sim_matrix.shape}")
    print(f"  Non-zero entries: {sim_matrix.nnz:,}")
    print(f"  Sparsity: {100 * sim_matrix.nnz / (sim_matrix.shape[0] ** 2):.2f}%")

    return sim_matrix


def _keep_topk_per_row(sparse_matrix, k):
    """
    Keep only top-K values per row in a sparse CSR matrix.
    Significantly reduces matrix size for large K.

    Parameters
    ----------
    sparse_matrix : csr_matrix
    k : int

    Returns
    -------
    csr_matrix with at most k non-zero per row
    """
    data = []
    indices = []
    indptr = [0]

    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix[i]
        if row.nnz == 0:
            indptr.append(indptr[-1])
            continue

        # Get indices of top-K values in this row
        row_data = row.data
        row_indices = row.indices

        if row.nnz <= k:
            top_indices = np.arange(row.nnz)
        else:
            top_indices = np.argpartition(row_data, -k)[-k:]
            # Sort descending by value
            top_indices = top_indices[np.argsort(row_data[top_indices])[::-1]]

        data.append(row_data[top_indices])
        indices.append(row_indices[top_indices])
        indptr.append(indptr[-1] + len(top_indices))

    data = np.concatenate(data)
    indices = np.concatenate(indices)

    return csr_matrix((data, indices, indptr), shape=sparse_matrix.shape)


def build_cb_model(products_df, max_features=10000, top_k=100):
    """
    Full pipeline: build TF-IDF vectors → cosine similarity → save.

    Parameters
    ----------
    products_df : pd.DataFrame
        Products data from data_loader.load_products().
    max_features : int
        TF-IDF vocabulary size.
    top_k : int
        Keep top-K similar items per product (reduces storage).

    Returns
    -------
    tuple: (tfidf_matrix, similarity_matrix, vectorizer)
    """
    print("=" * 50)
    print("Building Content-Based (CB) Model")
    print("=" * 50)

    # Step 1: Build documents
    print("\n[1/3] Building text documents...")
    documents, product_ids = build_documents(products_df)
    print(f"  Created {len(documents)} documents")

    # Step 2: TF-IDF vectorize
    print("\n[2/3] TF-IDF vectorization...")
    tfidf_matrix, vectorizer = build_tfidf(documents, max_features=max_features)

    # Step 3: Cosine similarity
    print("\n[3/3] Computing similarity...")
    sim_matrix = build_similarity(tfidf_matrix, top_k=top_k, threshold=None)

    return tfidf_matrix, sim_matrix, vectorizer


def save_model(tfidf_matrix, sim_matrix, vectorizer):
    """
    Save CB model outputs to models/ directory.

    Parameters
    ----------
    tfidf_matrix : csr_matrix
    sim_matrix : csr_matrix
    vectorizer : TfidfVectorizer
    """
    import pandas as pd

    print("\nSaving CB model outputs...")

    # Save sparse matrices
    save_npz(MODELS_DIR / "tfidf_matrix.npz", tfidf_matrix)
    print(f"  Saved: models/tfidf_matrix.npz")

    save_npz(MODELS_DIR / "item_similarity_cb.npz", sim_matrix)
    print(f"  Saved: models/item_similarity_cb.npz")

    # Save vectorizer
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved: models/tfidf_vectorizer.pkl")

    print("\nCB model complete!")


if __name__ == "__main__":
    # Import here to avoid circular import issues at module level
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_products

    # Load data
    products_df = load_products()

    # Build model
    tfidf_matrix, sim_matrix, vectorizer = build_cb_model(
        products_df, max_features=10000, top_k=100
    )

    # Save
    save_model(tfidf_matrix, sim_matrix, vectorizer)