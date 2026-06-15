"""
TF-IDF cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name_vi dùng word n-gram TF-IDF.
Giữ nguyên dấu tiếng Việt, không lowercase bừa bãi trước khi clean.
"""
import os
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.config import (
    CB_N_GRAM_RANGE, CB_MAX_FEATURES,
    CB_COUNT_N_GRAM_RANGE, CB_COUNT_MAX_FEATURES,
    CB_ALPHA, PROJECT_ROOT
)

# Pattern gom Nhóm đơn vị đo lường, khối lượng, dung tích và kích cỡ
_PATTERN_CLEAN = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:ct|count|mg|mcg|oz|fl\s*oz|fl|gallon|inch|in|pack|pk|ml|liter|lit|lít|l|lb|lbs|iu|i\.u\.?|loads|watt|cups|cup|sticks|g|kg|gr|grs|cm|mm)\b"
    r"|\b(?:size|cỡ)\s*\d+\b",
    re.IGNORECASE
)

# Cache 2 Pattern Regex của Stopwords để dùng lại cho mọi sản phẩm, không compile lại
_REGEX_WORD_STOPWORDS = None
_REGEX_SPECIAL_STOPWORDS = None


def _init_compiled_stopwords():
    """Khởi tạo và gộp toàn bộ stopwords thành 2 Pattern Regex tối ưu duy nhất."""
    global _REGEX_WORD_STOPWORDS, _REGEX_SPECIAL_STOPWORDS
    
    if _REGEX_WORD_STOPWORDS is not None:
        return

    path = os.path.join(PROJECT_ROOT, "vietnamese_stopwords.txt")
    if not os.path.exists(path):
        print(f"[WARN] Không tìm thấy {path}, bỏ qua stop words.")
        _REGEX_WORD_STOPWORDS = re.compile(r'$^')  # Match rỗng nếu không có file
        _REGEX_SPECIAL_STOPWORDS = re.compile(r'$^')
        return

    with open(path, "r", encoding="utf-8") as f:
        raw_words = [line.strip().lower() for line in f if line.strip()]
        # Loại bỏ các từ trùng lặp trong file txt của bạn
        unique_words = list(set(raw_words))
        # Sắp xếp từ dài trước, từ ngắn sau để tránh match thiếu cụm từ
        sorted_words = sorted(unique_words, key=len, reverse=True)

    word_patterns = []
    special_patterns = []

    for word in sorted_words:
        # Nếu là từ bình thường (chứa chữ/số) -> Dùng \b để bắt trọn vẹn từ
        if re.match(r'^\w', word) and re.search(r'\w$', word):
            word_patterns.append(re.escape(word))
        else:
            # Nếu là ký tự đặc biệt (như &) -> Không dùng \b
            special_patterns.append(re.escape(word))

    # Gộp thành 2 Regex Pattern lớn bằng toán tử | (OR)
    if word_patterns:
        _REGEX_WORD_STOPWORDS = re.compile(r'\b(' + '|'.join(word_patterns) + r')\b', re.IGNORECASE)
    else:
        _REGEX_WORD_STOPWORDS = re.compile(r'$^')

    if special_patterns:
        _REGEX_SPECIAL_STOPWORDS = re.compile(r'(' + '|'.join(special_patterns) + r')', re.IGNORECASE)
    else:
        _REGEX_SPECIAL_STOPWORDS = re.compile(r'$^')

# === TỐI ƯU HIỆU NĂNG: Khởi tạo luôn ở tầng Global, chạy 1 lần duy nhất khi import file ===
_init_compiled_stopwords()


def _clean_text_preprocessor(text):
    """Hàm tiền xử lý chuỗi: Tốc độ cao cực hạn nhờ loại bỏ hoàn toàn vòng lặp for."""
    if not isinstance(text, str):
        return ""

    # 1. Đưa về lowercase
    text = text.lower()

    # 2. Xóa sạch dung tích, kích thước dựa trên Regex định sẵn
    text = _PATTERN_CLEAN.sub("", text)

    # 3. Quét sạch toàn bộ Stopwords cực nhanh bằng Regex đã compile sẵn toàn cục
    text = _REGEX_SPECIAL_STOPWORDS.sub('', text)
    text = _REGEX_WORD_STOPWORDS.sub('', text)

    # 4. Loại bỏ các ký tự đặc biệt rác còn sót lại
    text = re.sub(r'[^\w\s]', ' ', text)

    # 5. Dọn dẹp khoảng trắng thừa phát sinh
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_product_vectors(text_data, ngram_range=CB_N_GRAM_RANGE, max_features=CB_MAX_FEATURES, analyzer='word'):
    """
    Xây dựng ma trận TF-IDF từ danh sách tên sản phẩm.
    """
    print(f"  TF-IDF ({analyzer}, ngram_range={ngram_range}, max_features={max_features})...")

    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer,
        preprocessor=_clean_text_preprocessor,
        stop_words=None,
    )

    tfidf_matrix = tfidf.fit_transform(text_data)
    print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")

    return tfidf_matrix, tfidf


def cb_similarity(product_vectors, product_a_idx, candidate_indices):
    """
    Tính cosine similarity giữa product_a và từng candidate — on-demand.
    """
    vec_a = product_vectors[product_a_idx]
    vecs_b = product_vectors[candidate_indices]

    dot_ab = vecs_b.dot(vec_a.T).toarray().ravel()
    return dot_ab


def build_count_vectors(text_data, ngram_range=CB_COUNT_N_GRAM_RANGE,
                        max_features=CB_COUNT_MAX_FEATURES,
                        analyzer='word'):
    """
    Xây dựng ma trận Count Vectorizer từ danh sách tên sản phẩm.
    """
    print(f"  CountVectorizer ({analyzer}, ngram_range={ngram_range}, "
          f"max_features={max_features})...")

    count_vec = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer,
        preprocessor=_clean_text_preprocessor,
        stop_words=None,
    )

    count_matrix = count_vec.fit_transform(text_data)
    print(f"    CountVectorizer matrix shape: {count_matrix.shape}")

    return count_matrix, count_vec


def cb_ensemble_similarity(product_vectors_tfidf, product_vectors_count,
                           product_a_idx, candidate_indices, alpha=CB_ALPHA):
    """
    Tính ensemble similarity kết hợp Count (Overlap Coefficient) + TF-IDF (Cosine).
    """
    # 1. Cosine similarity trên TF-IDF
    vec_a_tfidf = product_vectors_tfidf[product_a_idx]
    vecs_b_tfidf = product_vectors_tfidf[candidate_indices]
    sim_tfidf = vecs_b_tfidf.dot(vec_a_tfidf.T).toarray().ravel()

    # 2. Nhánh Count Vectorizer (Overlap Coefficient)
    vec_a_cnt = product_vectors_count[product_a_idx]
    vecs_b_cnt = product_vectors_count[candidate_indices]

    bin_a = vec_a_cnt.astype(bool).astype(np.float64)
    bin_b = vecs_b_cnt.astype(bool).astype(np.float64)

    # Giao = số từ trùng
    intersection = bin_b.dot(bin_a.T).toarray().ravel()

    # Số từ mỗi bên
    sum_a = bin_a.sum()
    sum_b = np.array(bin_b.sum(axis=1)).ravel()

    # Overlap Coefficient công thức mới: phủ định độ dài câu dài
    # === TRÁNH BUG: Ép kiểu .ravel() để mảng min_lengths phẳng hoàn toàn 1D ===
    min_lengths = np.minimum(sum_a, sum_b).ravel()

    sim_count = np.divide(intersection, min_lengths,
                          out=np.zeros_like(intersection, dtype=float),
                          where=min_lengths != 0)

    # 3. Kết hợp Ensemble
    final_sim = alpha * sim_count + (1.0 - alpha) * sim_tfidf
    return final_sim