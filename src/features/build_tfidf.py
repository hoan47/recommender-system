"""
Content-Based (CB) model: TF-IDF + Cosine Similarity.

Xây dựng ma trận item-item similarity dựa trên tên sản phẩm và department.
Được dùng làm:
  - Baseline model để so sánh
  - Bộ lọc loại bỏ sản phẩm thay thế (substitute) khỏi SPMI/KG recommendations
  - Fallback cho sản phẩm long-tail có ít dữ liệu co-occurrence

Phụ thuộc: src.utils.data_loader

Outputs:
  - models/tfidf_matrix.npz        - Ma trận TF-IDF sparse gốc
  - models/item_similarity_cb.npz  - Ma trận cosine similarity sparse
  - models/tfidf_vectorizer.pkl    - TfidfVectorizer đã fit
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Tạo thư mục models nếu chưa tồn tại
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_documents(products_df):
    """
    Tạo text documents cho TF-IDF từ metadata sản phẩm.

    Mỗi document = product_name + " " + department.
    Xử lý giá trị missing/NaN an toàn.

    Parameters
    ----------
    products_df : pd.DataFrame
        Với các cột: product_id, product_name, department

    Returns
    -------
    tuple: (documents, product_ids)
        documents: list các string
        product_ids: numpy array thẳng hàng với documents
    """
    import pandas as pd

    documents = []
    product_ids = []

    for _, row in products_df.iterrows():
        product_name = row.get("product_name", "")
        department = row.get("department", "")

        # Xử lý giá trị missing / NaN
        if pd.isna(product_name) or str(product_name).strip() == "":
            product_name = "unknown product"
        if pd.isna(department) or str(department).strip() == "":
            department = "unknown department"

        # Tạo document
        doc = f"{product_name} {department}"
        documents.append(doc)
        product_ids.append(row["product_id"])

    return documents, np.array(product_ids)


def build_tfidf(documents, max_features=10000):
    """
    Xây dựng ma trận TF-IDF từ text documents.

    Parameters
    ----------
    documents : list of str
        Text documents cho mỗi sản phẩm.
    max_features : int
        Số lượng TF-IDF features tối đa (kích thước vocabulary).

    Returns
    -------
    tuple: (tfidf_matrix, vectorizer)
        tfidf_matrix: scipy.sparse.csr_matrix (n_products x max_features)
        vectorizer: TfidfVectorizer đã fit
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,  # 1 + log(tf)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Kích thước vocabulary: {len(vectorizer.vocabulary_)}")

    return tfidf_matrix, vectorizer


def build_similarity(tfidf_matrix, top_k=None, threshold=None):
    """
    Tính cosine similarity và tùy chọn sparsify.

    Parameters
    ----------
    tfidf_matrix : scipy.sparse.csr_matrix
        TF-IDF vectors (n_products x n_features).
    top_k : int hoặc None
        Nếu set, chỉ giữ top-K sản phẩm tương tự nhất cho mỗi sản phẩm.
    threshold : float hoặc None
        Nếu set, chỉ giữ similarity trên ngưỡng này.

    Returns
    -------
    scipy.sparse.csr_matrix
        Cosine similarity (n_products x n_products), sparse format.
    """
    print("  Đang tính cosine similarity...")
    # cosine_similarity trên sparse input trả về dense array
    # Ta tính từng dòng và giữ sparse để tiết kiệm RAM
    n = tfidf_matrix.shape[0]
    sim_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

    # Đảm bảo là CSR format
    if not isinstance(sim_matrix, csr_matrix):
        sim_matrix = sim_matrix.tocsr()

    # Gán đường chéo về 0 (sản phẩm luôn giống chính nó)
    sim_matrix.setdiag(0)
    sim_matrix.eliminate_zeros()

    # Áp threshold nếu có
    if threshold is not None:
        print(f"  Đang lọc similarity > {threshold}...")
        sim_matrix.data[sim_matrix.data <= threshold] = 0
        sim_matrix.eliminate_zeros()

    # Giữ top-K mỗi dòng nếu có
    if top_k is not None:
        print(f"  Đang giữ top-{top_k} mỗi sản phẩm...")
        sim_matrix = _keep_topk_per_row(sim_matrix, top_k)

    print(f"  Similarity matrix shape: {sim_matrix.shape}")
    print(f"  Non-zero entries: {sim_matrix.nnz:,}")
    print(f"  Độ sparsity: {100 * sim_matrix.nnz / (sim_matrix.shape[0] ** 2):.2f}%")

    return sim_matrix


def _keep_topk_per_row(sparse_matrix, k):
    """
    Chỉ giữ top-K giá trị mỗi dòng trong sparse CSR matrix.
    Giảm đáng kể kích thước ma trận khi K lớn.

    Parameters
    ----------
    sparse_matrix : csr_matrix
    k : int

    Returns
    -------
    csr_matrix với tối đa k non-zero mỗi dòng
    """
    data = []
    indices = []
    indptr = [0]

    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix[i]
        if row.nnz == 0:
            indptr.append(indptr[-1])
            continue

        # Lấy index của top-K giá trị trong dòng này
        row_data = row.data
        row_indices = row.indices

        if row.nnz <= k:
            top_indices = np.arange(row.nnz)
        else:
            top_indices = np.argpartition(row_data, -k)[-k:]
            # Sắp xếp giảm dần theo giá trị
            top_indices = top_indices[np.argsort(row_data[top_indices])[::-1]]

        data.append(row_data[top_indices])
        indices.append(row_indices[top_indices])
        indptr.append(indptr[-1] + len(top_indices))

    data = np.concatenate(data)
    indices = np.concatenate(indices)

    return csr_matrix((data, indices, indptr), shape=sparse_matrix.shape)


def build_cb_model(products_df, max_features=10000, top_k=100):
    """
    Pipeline đầy đủ: tạo TF-IDF vectors → cosine similarity → lưu.

    Parameters
    ----------
    products_df : pd.DataFrame
        Dữ liệu sản phẩm từ data_loader.load_products().
    max_features : int
        Kích thước vocabulary TF-IDF.
    top_k : int
        Giữ top-K similar items mỗi sản phẩm (giảm dung lượng lưu trữ).

    Returns
    -------
    tuple: (tfidf_matrix, similarity_matrix, vectorizer)
    """
    print("=" * 50)
    print("Xây dựng Content-Based (CB) Model")
    print("=" * 50)

    # Bước 1: Tạo documents
    print("\n[1/3] Đang tạo text documents...")
    documents, product_ids = build_documents(products_df)
    print(f"  Đã tạo {len(documents)} documents")

    # Bước 2: TF-IDF vectorize
    print("\n[2/3] TF-IDF vectorization...")
    tfidf_matrix, vectorizer = build_tfidf(documents, max_features=max_features)

    # Bước 3: Cosine similarity
    print("\n[3/3] Đang tính similarity...")
    sim_matrix = build_similarity(tfidf_matrix, top_k=top_k, threshold=None)

    return tfidf_matrix, sim_matrix, vectorizer


def save_model(tfidf_matrix, sim_matrix, vectorizer):
    """
    Lưu CB model outputs vào thư mục models/.

    Parameters
    ----------
    tfidf_matrix : csr_matrix
    sim_matrix : csr_matrix
    vectorizer : TfidfVectorizer
    """
    print("\nĐang lưu CB model outputs...")

    # Lưu sparse matrices
    save_npz(MODELS_DIR / "tfidf_matrix.npz", tfidf_matrix)
    print(f"  Đã lưu: models/tfidf_matrix.npz")

    save_npz(MODELS_DIR / "item_similarity_cb.npz", sim_matrix)
    print(f"  Đã lưu: models/item_similarity_cb.npz")

    # Lưu vectorizer
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Đã lưu: models/tfidf_vectorizer.pkl")

    print("\nCB model hoàn tất!")


if __name__ == "__main__":
    # Import ở đây để tránh circular import ở module level
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.data_loader import load_products

    # Load dữ liệu
    products_df = load_products()

    # Xây dựng model
    tfidf_matrix, sim_matrix, vectorizer = build_cb_model(
        products_df, max_features=10000, top_k=100
    )

    # Lưu
    save_model(tfidf_matrix, sim_matrix, vectorizer)