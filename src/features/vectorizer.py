"""
TF-IDF cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name_vi dùng word n-gram TF-IDF.
Giữ nguyên dấu tiếng Việt, không lowercase.
"""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CB_N_GRAM_RANGE, CB_MAX_FEATURES, CB_ANALYZER

# Pattern gom Nhóm đơn vị đo lường và Nhóm kích cỡ (size/cỡ)
_PATTERN_CLEAN = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:ct|count|mg|mcg|oz|fl\s*oz|fl|gallon|inch|in|pack|pk|ml|liter|lit|lít|l|lb|lbs|iu|i\.u\.?|loads|watt|cups|cup|sticks)\b"
    r"|\b(?:size|cỡ)\s*\d+\b",
    re.IGNORECASE
)


def _clean_product_name_vi(text: str) -> str:
    """
    Làm sạch tên sản phẩm tiếng Việt:
      - Giữ nguyên dấu tiếng Việt, không lowercase
      - Xóa các cụm dung tích, quy cách (số + đơn vị, size/cỡ + số)
      - Dọn dẹp ký tự thừa phát sinh sau khi xóa
    """
    if not isinstance(text, str):
        return ''

    # 1. Xóa sạch các cụm dung tích, quy cách bằng Regex
    text = _PATTERN_CLEAN.sub('', text)

    # 2. Dọn dẹp các ký tự thừa phát sinh sau khi xóa số
    text = re.sub(r'\s*-\s*$', '', text)          # Xóa dấu gạch ngang ở cuối câu
    text = re.sub(r'\s*-\s*(?=\s)', ' ', text)    # Xóa dấu gạch ngang bị cô lập giữa câu
    text = re.sub(r'\(\s*\)', '', text)            # Xóa dấu ngoặc đơn rỗng dạng ()
    text = re.sub(r'\s+', ' ', text).strip()      # Gom khoảng trắng thừa thành khoảng trắng đơn

    return text


def build_product_vectors(products_df, ngram_range=None, max_features=None, analyzer=None):
    """
    Vector hóa sản phẩm (word n-gram TF-IDF trên product_name_vi).

    Args:
        products_df: DataFrame [product_id, product_name_vi, ...]
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
    print(f"Đang vector hóa {n_products} sản phẩm (tiếng Việt)...")

    # Làm sạch tên sản phẩm tiếng Việt — giữ nguyên dấu, không lowercase
    text_data = products_df['product_name_vi'].fillna('').apply(_clean_product_name_vi)

    # TF-IDF với word n-gram (không dùng stop words vì là tiếng Việt)
    print(f"  TF-IDF ({analyzer}, ngram_range={ngram_range}, max_features={max_features})...")
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer,
        # Không dùng stop words — tiếng Việt không có file stop words chuẩn
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