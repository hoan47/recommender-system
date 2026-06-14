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

# Pattern gom Nhóm đơn vị đo lường, khối lượng, dung tích và kích cỡ
# Cải tiến: Thêm g, kg, gr, cm, mm và xử lý chặt chẽ hơn các ký tự dính liền
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
        # (ví dụ: 'tự nhiên' trước 'tự', 'chính hãng' trước 'hãng')
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

    # 3. Loại bỏ các ký tự đặc biệt rác hay có trong tên sản phẩm (giữ lại dấu cách)
    text = re.sub(r'[^\w\s]', ' ', text)

    # 4. XỬ LÝ TRIỆT ĐỂ STOP WORDS (Bao gồm cả từ ghép như "chính hãng", "cao cấp")
    #    Xóa cụm dài (2-3 từ) trước, từ ngắn sau — tránh mất context
    stop_words = _load_vietnamese_stopwords()
    for word in stop_words:
        # Sử dụng \b để chỉ xóa khi nó là một từ trọn vẹn, tránh xóa nhầm một phần của từ khác
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)

    # 5. Dọn dẹp khoảng trắng thừa phát sinh sau khi xóa từ
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_product_vectors(text_data, ngram_range=CB_N_GRAM_RANGE, max_features=CB_MAX_FEATURES, analyzer=CB_ANALYZER):
    """
    Xây dựng ma trận TF-IDF từ danh sách tên sản phẩm.
    """
    print(f"  TF-IDF ({analyzer}, ngram_range={ngram_range}, max_features={max_features})...")

    # Vì ta đã xử lý stop_words thủ công ở tầng `preprocessor` rất sạch sẽ,
    # nên ta đặt stop_words của Sklearn là None để tránh xung đột hoặc lỗi cảnh báo.
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