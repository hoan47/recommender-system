"""
TF-IDF cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name.
"""
import os
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CB_N_GRAM_RANGE, CB_MAX_FEATURES, PROJECT_ROOT

# ============================================================
# Regex patterns cho làm sạch tên sản phẩm (3 bước)
# ============================================================

# Bước 1: Xoá các từ chỉ thông số đứng trước số, VD: "size 12", "spf 50", "sheets 100"
_PATTERN_PARAM_BEFORE_NUM = re.compile(
    r'\b(size|spf|posted|uploaded|bags|buns|tablets|sheets)\s*\d+(\.\d+)?\b',
    re.IGNORECASE
)

# Bước 2: Xoá số + đơn vị đo lường, VD: "100 mg", "2 oz", "50 ct"
_PATTERN_NUM_UNIT = re.compile(
    r'\b\d+(\.\d+)?\s*'
    r'(mg|mcg|g|iu|oz|fl|floz|gallon|gal|qt|liter|l|ml|lb|lbs|ct|count|'
    r'pk|pack|loads|sheets|sticks|pieces|slices|tablets|refills|in|inch|'
    r'ply|calories?|calorie|t|m|year|months?|x)\b',
    re.IGNORECASE
)

# Bước 3: Xoá toàn bộ chữ số và ký tự đặc biệt còn sót lại
_PATTERN_NON_ALPHA = re.compile(r'[^a-zA-Z\s]+')


def _clean_product_name(name: str) -> str:
    """
    Làm sạch tên sản phẩm qua 3 bước regex:
      1. Xoá từ chỉ thông số đứng trước số (size 12, spf 50, ...)
      2. Xoá số + đơn vị đo lường (100 mg, 2 oz, ...)
      3. Xoá tất cả chữ số và ký tự đặc biệt còn sót
    Kết quả: chỉ còn lại các từ thuần tuý (a-zA-Z), lowercase.
    """
    if not name or not isinstance(name, str):
        return ''
    # Bước 1
    text = _PATTERN_PARAM_BEFORE_NUM.sub(' ', name)
    # Bước 2
    text = _PATTERN_NUM_UNIT.sub(' ', text)
    # Bước 3
    text = _PATTERN_NON_ALPHA.sub(' ', text)
    # Chuẩn hoá khoảng trắng
    text = ' '.join(text.split())
    return text.lower()


def _load_stop_words():
    """
    Đọc file english_stopwords.txt.
    Mỗi từ 1 dòng, dòng bắt đầu bằng # được bỏ qua.
    Trả về list để tương thích với sklearn TfidfVectorizer.
    """
    stop_words = set()
    
    # Đọc file stop words tiếng Anh custom
    en_file = os.path.join(PROJECT_ROOT, "english_stopwords.txt")
    if os.path.exists(en_file):
        with open(en_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith('#'):
                    stop_words.add(word)
    
    return list(stop_words)

def build_product_vectors(products_df, ngram_range=None, max_features=None):
    """
    Vector hóa sản phẩm (TF-IDF trên product_name).

    Args:
        products_df: DataFrame [product_id, product_name, ...]
        ngram_range: tuple (min_n, max_n) cho TF-IDF
        max_features: int, max features cho TF-IDF

    Returns:
        product_vectors: sparse.csr_matrix shape (n_products, D)
        vectorizer: TfidfVectorizer đã fit
    """
    if ngram_range is None:
        ngram_range = CB_N_GRAM_RANGE
    if max_features is None:
        max_features = CB_MAX_FEATURES
    
    n_products = len(products_df)
    print(f"Đang vector hóa {n_products} sản phẩm...")
    
    # Chỉ dùng product_name (EN) — làm sạch trước khi vector hóa
    text_data = products_df['product_name'].fillna('').apply(_clean_product_name)
    
    # --- TF-IDF ---
    print("  TF-IDF...")
    stop_words = _load_stop_words()
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        stop_words=stop_words,
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