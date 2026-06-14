"""
TF-IDF cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name dùng character n-gram TF-IDF.
"""
import os
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CB_N_GRAM_RANGE, CB_MAX_FEATURES, CB_ANALYZER, PROJECT_ROOT

# Regex cleaning — 3 bước tối ưu cho tên sản phẩm Instacart
_PATTERN_SPEC_PARAM = re.compile(
    r'\b(size|spf|posted|uploaded|bags|buns|tablets|sheets)\s*\d+(\.\d+)?\b',
    re.IGNORECASE
)
_PATTERN_UNIT = re.compile(
    r'\b\d+(\.\d+)?\s*(mg|mcg|g|iu|oz|fl|floz|gallon|gal|qt|liter|l|ml|lb|lbs|ct|count|pk|pack|loads|sheets|sticks|pieces|slices|tablets|refills|in|inch|ply|calories?|calorie|t|m|year|months?|x)\b',
    re.IGNORECASE
)
_PATTERN_NON_ALPHA = re.compile(r'[^a-zA-Z\s]+')

# === Load stop words từ file english_stopwords.txt ===
_STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "english_stopwords.txt")
_ENGLISH_STOP_WORDS = None


def _load_stopwords():
    """Load stop words từ file english_stopwords.txt."""
    global _ENGLISH_STOP_WORDS
    if _ENGLISH_STOP_WORDS is not None:
        return _ENGLISH_STOP_WORDS
    if not os.path.exists(_STOP_WORDS_PATH):
        print(f"[WARN] Không tìm thấy {_STOP_WORDS_PATH}, bỏ qua stop words.")
        _ENGLISH_STOP_WORDS = []
        return _ENGLISH_STOP_WORDS
    with open(_STOP_WORDS_PATH, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    _ENGLISH_STOP_WORDS = words
    print(f"  Đã load {len(words)} stop words từ {_STOP_WORDS_PATH}")
    return _ENGLISH_STOP_WORDS


def _clean_product_name(name: str) -> str:
    """
    Làm sạch tên sản phẩm với 3 bước regex:
      1. Xoá từ chỉ thông số đứng trước số (vd: "size 12", "bags 10")
      2. Xoá đơn vị đo lường đứng sau số (vd: "16 oz", "2 lb")
      3. Xoá tất cả ký tự số và ký tự đặc biệt còn sót lại
    """
    if not name or not isinstance(name, str):
        return ''
    # Bước 1: xoá từ chỉ thông số + số (vd: "size 12" → "")
    text = _PATTERN_SPEC_PARAM.sub(' ', name)
    # Bước 2: xoá số + đơn vị (vd: "16 oz" → "")
    text = _PATTERN_UNIT.sub(' ', text)
    # Bước 3: xoá ký tự đặc biệt, giữ lại a-z + space
    text = _PATTERN_NON_ALPHA.sub(' ', text)
    # Chuẩn hoá khoảng trắng + lowercase
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

    # Load stop words (sẽ dùng khi analyzer='word', bỏ qua khi analyzer='char')
    stop_words = _load_stopwords()

    # TF-IDF với character n-gram
    print(f"  TF-IDF ({analyzer}, ngram_range={ngram_range}, max_features={max_features})...")
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer,
        stop_words=stop_words if stop_words else None,
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