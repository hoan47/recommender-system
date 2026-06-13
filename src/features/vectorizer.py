"""
TF-IDF cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name.
"""
import os
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import CB_N_GRAM_RANGE, CB_MAX_FEATURES, PROJECT_ROOT


def _load_stop_words():
    """
    Load stop words: sklearn's ENGLISH_STOP_WORDS + file english_stopwords.txt.
    Mỗi từ 1 dòng, dòng bắt đầu bằng # được bỏ qua.
    Trả về list để tương thích với sklearn TfidfVectorizer.
    """
    stop_words = set(ENGLISH_STOP_WORDS)
    
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
    
    # Chỉ dùng product_name (EN)
    text_data = products_df['product_name'].fillna('')
    
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