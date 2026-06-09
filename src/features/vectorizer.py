"""
TF-IDF + One-hot cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name, aisle_id, department_id.
"""
import os
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import OneHotEncoder

from src.config import CB_N_GRAM_RANGE, CB_MAX_FEATURES, PROJECT_ROOT


def _load_stop_words():
    """
    Load stop words: dung sklearn's ENGLISH_STOP_WORDS + tu custom trong file.
    File english_stopwords.txt: them tu moi vao day, moi tu 1 dong.
    """
    stop_words = set(ENGLISH_STOP_WORDS)
    stop_file = os.path.join(PROJECT_ROOT, "english_stopwords.txt")
    if os.path.exists(stop_file):
        with open(stop_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith('#'):
                    stop_words.add(word)
    return stop_words


def build_product_vectors(products_df, ngram_range=None, max_features=None):
    """
    Vector hóa sản phẩm:
      - TF-IDF trên product_name (unigram + bigram)
      - One-hot aisle_id
      - One-hot department_id
    Ghép dọc 3 vector lại bằng hstack.
    
    Args:
        products_df: DataFrame [product_id, product_name, aisle_id, department_id, ...]
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
    
    # --- TF-IDF trên product_name ---
    print("  TF-IDF trên product_name...")
    stop_words = _load_stop_words()
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        stop_words=stop_words,
    )
    tfidf_matrix = tfidf.fit_transform(products_df['product_name'].fillna(''))
    print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # --- One-hot aisle_id ---
    print("  One-hot aisle_id...")
    aisle_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    aisle_matrix = aisle_encoder.fit_transform(products_df[['aisle_id']])
    print(f"    Aisle one-hot shape: {aisle_matrix.shape}")
    
    # --- One-hot department_id ---
    print("  One-hot department_id...")
    dept_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    dept_matrix = dept_encoder.fit_transform(products_df[['department_id']])
    print(f"    Department one-hot shape: {dept_matrix.shape}")
    
    # --- Ghép dọc ---
    print("  Ghép vectors...")
    product_vectors = sparse.hstack([
        tfidf_matrix,
        aisle_matrix,
        dept_matrix
    ], format='csr')
    print(f"    Final product vectors shape: {product_vectors.shape}")
    
    # Lưu vectorizer và encoders vào attribute để dùng sau (nếu cần)
    product_vectors._tfidf = tfidf
    product_vectors._aisle_encoder = aisle_encoder
    product_vectors._dept_encoder = dept_encoder
    
    return product_vectors, tfidf


def cb_similarity(product_vectors, product_a_idx, candidate_indices):
    """
    Tính cosine similarity giữa product_a và từng candidate.
    Chỉ tính on-demand, không pre-compute full matrix.
    
    Args:
        product_vectors: sparse.csr_matrix (n_products, D)
        product_a_idx: int, index của product A
        candidate_indices: list[int], indices của các candidate
    
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