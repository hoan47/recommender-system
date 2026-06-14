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
from sklearn.preprocessing import normalize

from src.config import (
    CB_N_GRAM_RANGE, CB_MAX_FEATURES,
    CB_COUNT_N_GRAM_RANGE, CB_COUNT_MAX_FEATURES,
    CB_ALPHA, CB_METRIC, PROJECT_ROOT
)

# Pattern gom Nhóm đơn vị đo lường, khối lượng, dung tích và kích cỡ
_PATTERN_CLEAN = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:ct|count|mg|mcg|oz|fl\s*oz|fl|gallon|inch|in|pack|pk|ml|liter|lit|lít|l|lb|lbs|iu|i\.u\.?|loads|watt|cups|cup|sticks|g|kg|gr|grs|cm|mm)\b"
    r"|\b(?:size|cỡ)\s*\d+\b",
    re.IGNORECASE
)

# Biến global để cache stop words, tránh đọc file nhiều lần khi gọi hàm preprocessor
_VIETNAMESE_STOPWORDS = None


def _load_vietnamese_stopwords():
    """Load stop words tiếng Việt từ file vietnamese_stopwords.txt."""
    global _VIETNAMESE_STOPWORDS
    if _VIETNAMESE_STOPWORDS is not None:
        return _VIETNAMESE_STOPWORDS

    path = os.path.join(PROJECT_ROOT, "vietnamese_stopwords.txt")
    if not os.path.exists(path):
        print(f"[WARN] Không tìm thấy {path}, bỏ qua stop words.")
        _VIETNAMESE_STOPWORDS = []
        return _VIETNAMESE_STOPWORDS

    with open(path, "r", encoding="utf-8") as f:
        # Sắp xếp stop words theo chiều dài giảm dần để khi thay thế không bị đè
        words = [line.strip().lower() for line in f if line.strip()]
        _VIETNAMESE_STOPWORDS = sorted(words, key=len, reverse=True)
    return _VIETNAMESE_STOPWORDS


def _clean_text_preprocessor(text):
    """Hàm tiền xử lý chuỗi: xóa sạch dung tích/quy cách rác và stop words từ ghép."""
    if not isinstance(text, str):
        return ""

    # 1. Đưa về lowercase trước để đồng bộ cho Regex và Stopwords
    text = text.lower()

    # 2. Xóa sạch dung tích, kích thước dựa trên Regex
    text = _PATTERN_CLEAN.sub("", text)

    # 3. ĐƯA XỬ LÝ STOP WORDS LÊN TRƯỚC PUNCTUATION
    #    Xóa cụm dài (2-3 từ) trước, từ ngắn sau — tránh mất context
    #    Đặt trước punctuation để stop words đặc biệt (vd: &) không bị xóa nhầm
    stop_words = _load_vietnamese_stopwords()
    for word in stop_words:
        # Nếu stop word là ký tự chữ -> dùng \b để match từ trọn vẹn
        # Nếu stop word là ký tự đặc biệt (vd: &) -> không dùng \b kẻo lỗi Regex
        if re.match(r'^\w', word) and re.search(r'\w$', word):
            text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
        else:
            text = re.sub(re.escape(word), '', text)

    # 4. Loại bỏ các ký tự đặc biệt rác còn sót lại (giữ lại dấu cách)
    text = re.sub(r'[^\w\s]', ' ', text)

    # 5. Dọn dẹp khoảng trắng thừa phát sinh sau khi xóa từ
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_product_vectors(text_data, ngram_range=CB_N_GRAM_RANGE, max_features=CB_MAX_FEATURES, analyzer='word'):
    """
    Xây dựng ma trận TF-IDF từ danh sách tên sản phẩm.
    """
    print(f"  TF-IDF ({analyzer}, ngram_range={ngram_range}, max_features={max_features})...")

    # Đặt stop_words của Sklearn là None để tránh xung đột hoặc lỗi cảnh báo.
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer,
        preprocessor=_clean_text_preprocessor,
        stop_words=None,
    )

    tfidf_matrix = tfidf.fit_transform(text_data)
    print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")

    product_vectors = tfidf_matrix

    return product_vectors, tfidf


def cb_similarity(product_vectors, product_a_idx, candidate_indices):
    """
    Tính cosine similarity giữa product_a và từng candidate — on-demand.
    """
    vec_a = product_vectors[product_a_idx]
    vecs_b = product_vectors[candidate_indices]

    # Tính toán trực tiếp trên Ma trận thưa (Sparse Matrix) để tối ưu tốc độ
    dot_ab = vecs_b.dot(vec_a.T).toarray().ravel()
    return dot_ab


def build_count_vectors(text_data, ngram_range=CB_COUNT_N_GRAM_RANGE,
                        max_features=CB_COUNT_MAX_FEATURES,
                        analyzer='word'):
    """
    Xây dựng ma trận Count Vectorizer (L2-normalized) từ danh sách tên sản phẩm.
    Chuẩn hóa L2 để cosine similarity có ý nghĩa khi dùng raw count.
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
    # KHÔNG normalize — giữ raw count để Jaccard hoạt động đúng
    # Cosine similarity sẽ normalize on-the-fly trong cb_ensemble_similarity
    print(f"    CountVectorizer matrix shape: {count_matrix.shape}")

    return count_matrix, count_vec


def cb_ensemble_similarity(product_vectors_tfidf, product_vectors_count,
                           product_a_idx, candidate_indices, alpha=CB_ALPHA,
                           metric=CB_METRIC):
    """
    Tính ensemble similarity kết hợp Count (Jaccard/Cosine) + TF-IDF (Cosine).

    Args:
        product_vectors_tfidf: sparse CSR matrix từ TF-IDF
        product_vectors_count: sparse CSR matrix từ Count Vectorizer (raw count, chưa norm)
        product_a_idx: int, index của product đầu vào
        candidate_indices: list[int], indices của các candidate
        alpha: float, trọng số Count Vectorizer (0 = chỉ TF-IDF, 1 = chỉ Count)
        metric: str, 'jaccard' hoặc 'cosine' cho nhánh Count

    Returns:
        np.ndarray: mảng similarity scores cho từng candidate
    """
    # 1. Cosine similarity trên TF-IDF (luôn giữ)
    vec_a_tfidf = product_vectors_tfidf[product_a_idx]
    vecs_b_tfidf = product_vectors_tfidf[candidate_indices]
    sim_tfidf = vecs_b_tfidf.dot(vec_a_tfidf.T).toarray().ravel()

    # 2. Nhánh Count Vectorizer
    vec_a_cnt = product_vectors_count[product_a_idx]
    vecs_b_cnt = product_vectors_count[candidate_indices]

    if metric == 'jaccard':
        # Nhị phân hóa: chỉ cần biết từ có xuất hiện hay không
        bin_a = vec_a_cnt.astype(bool).astype(np.float64)
        bin_b = vecs_b_cnt.astype(bool).astype(np.float64)

        # Giao = số từ trùng
        intersection = bin_b.dot(bin_a.T).toarray().ravel()

        # Số từ mỗi bên
        sum_a = bin_a.sum()
        sum_b = np.array(bin_b.sum(axis=1)).ravel()

        # Hợp = |A| + |B| - |A∩B|
        union = sum_a + sum_b - intersection

        # Jaccard = Intersection / Union (tránh chia 0)
        sim_count = np.divide(intersection, union,
                              out=np.zeros_like(intersection, dtype=float),
                              where=union != 0)
    else:
        # Chuẩn hóa L2 trực tiếp on-the-fly cho kịch bản test Cosine
        norm_a = np.sqrt(vec_a_cnt.multiply(vec_a_cnt).sum())
        norm_b = np.sqrt(np.array(vecs_b_cnt.multiply(vecs_b_cnt).sum(axis=1)).ravel())

        dot_product = vecs_b_cnt.dot(vec_a_cnt.T).toarray().ravel()
        denominator = norm_a * norm_b

        # Tránh lỗi chia cho 0 nếu gặp chuỗi rỗng
        sim_count = np.divide(dot_product, denominator,
                              out=np.zeros_like(dot_product, dtype=float),
                              where=denominator != 0)

    # 3. Kết hợp Ensemble
    final_sim = alpha * sim_count + (1.0 - alpha) * sim_tfidf
    return final_sim
