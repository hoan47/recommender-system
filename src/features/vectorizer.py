"""
TF-IDF cho CB Diversity Filter.
Vector hóa sản phẩm dựa trên product_name_vi dùng word n-gram TF-IDF.
Giữ nguyên dấu tiếng Việt, không lowercase bừa bãi trước khi clean.
"""
import os
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CB_N_GRAM_RANGE, CB_MAX_FEATURES, CB_ANALYZER, PROJECT_ROOT

# Pattern gom Nhóm đơn vị đo lường và Nhóm kích cỡ (size/cỡ)
_PATTERN_CLEAN = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:ct|count|mg|mcg|oz|fl\s*oz|fl|gallon|inch|in|pack|pk|ml|liter|lit|lít|l|lb|lbs|iu|i\.u\.?|loads|watt|cups|cup|sticks)\b"
    r"|\b(?:size|cỡ)\s*\d+\b",
    re.IGNORECASE
)


def _clean_text_preprocessor(text):
    """Hàm tiền xử lý chuỗi: xóa sạch dung tích/quy cách rác trước khi đưa vào TF-IDF."""
    if not isinstance(text, str):
        return ""

    # 1. Xóa sạch dung tích, kích thước dựa trên Regex đã chốt
    text = _PATTERN_CLEAN.sub("", text)

    # 2. Dọn dẹp hậu quả (dấu gạch ngang mồ côi, khoảng trắng kép, ký tự kích thước rác)
    text = re.sub(r'\s*-\s*$', '', text)          # Cuối câu
    text = re.sub(r'\s*-\s*(?=\s)', ' ', text)    # Giữa câu
    text = re.sub(r'\([\s*xX/,\.-]*\)', '', text)  # Dọn sạch dấu ngoặc chứa ký tự nhân/chia mồ côi
    text = re.sub(r'\s+', ' ', text).strip()      # Gom khoảng trắng thừa

    # 3. Đưa về lowercase đồng bộ cho mô hình TF-IDF dễ tính toán
    return text.lower()


def _load_vietnamese_stopwords():
    """Load stop words tiếng Việt từ file vietnamese_stopwords.txt."""
    path = os.path.join(PROJECT_ROOT, "vietnamese_stopwords.txt")
    if not os.path.exists(path):
        print(f"[WARN] Không tìm thấy {path}, bỏ qua stop words.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        # Đưa stop words về chữ thường để khớp với văn bản sau khi preprocessor hạ case
        return [line.strip().lower() for line in f if line.strip()]


def build_product_vectors(text_data, ngram_range=CB_N_GRAM_RANGE, max_features=CB_MAX_FEATURES, analyzer=CB_ANALYZER):
    """
    Xây dựng ma trận TF-IDF từ danh sách tên sản phẩm.
    """
    stop_words = _load_vietnamese_stopwords()

    # Nếu analyzer là dựa trên ký tự (char/char_wb), không áp dụng stop_words dạng từ đơn để tránh lỗi Sklearn
    if analyzer in ['char', 'char_wb']:
        actual_stop_words = None
        print(f"[INFO] Analyzer là '{analyzer}', tự động bỏ qua list stop_words dạng từ.")
    else:
        actual_stop_words = stop_words if stop_words else None

    print(f"  TF-IDF ({analyzer}, ngram_range={ngram_range}, max_features={max_features})...")

    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer,
        preprocessor=_clean_text_preprocessor,  # ĐÃ KÍCH HOẠT: Gọi hàm xóa rác ở đây
        stop_words=actual_stop_words,
    )

    tfidf_matrix = tfidf.fit_transform(text_data)
    print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")

    product_vectors = tfidf_matrix
    product_vectors._tfidf = tfidf

    return product_vectors, tfidf


def cb_similarity(product_vectors, product_a_idx, candidate_indices):
    """
    Tính cosine similarity giữa product_a và từng candidate — on-demand.
    """
    vec_a = product_vectors[product_a_idx]
    vecs_b = product_vectors[candidate_indices]

    # Cosine similarity thủ công cho sparse matrix
    dot_ab = vecs_b.dot(vec_a.T).toarray().ravel()
    return dot_ab