"""
TF-IDF cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name dùng character n-gram TF-IDF.
"""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CB_N_GRAM_RANGE, CB_MAX_FEATURES, CB_ANALYZER

# Regex đơn giản: chỉ giữ a-z, space (bỏ số, ký tự đặc biệt)
_PATTERN_CLEAN = re.compile(r'[^a-zA-Z\s]+')


def _clean_product_name(name: str) -> str:
    """
    Làm sạch tên sản phẩm đơn giản:
      - Xoá tất cả ký tự không phải a-zA-Z hoặc space
      - Lowercase
      - Chuẩn hoá khoảng trắng
    """
    if not name or not isinstance(name, str):
        return ''
    text = _PATTERN_CLEAN.sub(' ', name)
    text = ' '.join(text.split())
    return text.lower()


def build_product_vectors(products_df, ngram_range=None, max_features=None, analyzer=None):
    """
    Vector hóa sản phẩm (character n-gram TF-IDF trên product_name).

    Args:
        products_df: DataFrame [product_id, product_name, ...]
        ngram_range: tuple (min_n, max_n) cho TF-IDF
        max_features: int, max features cho TF-IDF
        analyzer: 'char' hoặc 'word'

    Returns:
        product_vectors: sparse.csr_matrix shape (n_products, D)
        vectorizer: TfidfVectorizer đã fit
    """
    if ngram_range is None:
        ngram_range = CB_N_GRAM_RANGE
    if max_features is None:
        max_features = CB_MAX_FEATURES
    if analyzer is None:
        analyzer = CB_ANALYZER

    n_products = len(products_df)
    print(f"Đang vector hóa {n_products} sản phẩm...")

    # Làm sạch đơn giản — chỉ lowercase + loại ký tự đặc biệt
    text_data = products_df['product_name'].fillna('').apply(_clean_product_name)

    # TF-IDF với character n-gram
    print(f"  TF-IDF ({analyzer}, ngram_range={ngram_range}, max_features={max_features})...")
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer,  # 'char' thay vì 'word'
        stop_words=None,    # không cần stop words cho char n-gram
    )
    tfidf_matrix = tfidf.fit_transform(text_data)
    print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")

    product_vectors = tfidf_matrix

    # Lưu vectorizer vào attribute để dùng sau (nếu cần)
    product_vectors._tfidf = tfidf

    return product_vectors, tfidf


def cb_similarity(product_vectors, product_a_idx, candidate_indices):
    """
    Tính cosine similarity giữa product_a và từng candidate — on-demand.

    Args:
        product_vectors: sparse.csr_matrix (n_products, D)
        product_a_idx: int — index của product A
        candidate_indices: list[int] — indices của các candidate

    Returns:
        numpy array shape (len(candidate_indices),) — similarity scores [0,1]
    """
    vec_a = product_vectors[product_a_idx]
    vecs_b = product_vectors[candidate_indices]

    # Cosine similarity thủ công cho sparse matrix
    dot_ab = vecs_b.dot(vec_a.T).toarray().flatten()
    norm_a = np.sqrt(vec_a.dot(vec_a.T).toarray()[0, 0])
    norms_b = np.sqrt((vecs_b.multiply(vecs_b)).sum(axis=1)).A1

    denom = norm_a * norms_b
    denom[denom == 0] = 1e-9  # tránh chia 0

    similarities = dot_ab / denom
    return np.clip(similarities, 0, 1)