"""
Content-Based (CB) model: TF-IDF + Cosine Similarity.

Pipeline TF-IDF và cosine similarity, dùng numpy + scipy.sparse.

Pipeline:
  1. Tokenize + xây dựng vocabulary (unigram + bigram, top max_features theo DF)
  2. TF (sublinear: 1 + log(tf)) × IDF (smooth) → L2 normalize
  3. Cosine similarity = chunked dot product giữa các dòng đã L2-normalize

Xây dựng ma trận item-item similarity dựa trên tên sản phẩm và department.
Được dùng làm:
  - Baseline model để so sánh
  - Bộ lọc loại bỏ sản phẩm thay thế (substitute) khỏi SPMI/KG recommendations
  - Fallback cho sản phẩm long-tail có ít dữ liệu co-occurrence

Phụ thuộc: src.utils.data_loader

Outputs:
  - models/tfidf_matrix.npz        - Ma trận TF-IDF sparse gốc
  - models/item_similarity_cb.npz  - Ma trận cosine similarity sparse
  - models/tfidf_vocab.json        - Vocabulary đã xây dựng
"""

import gc
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, save_npz

from src.config import (
    MODELS_DIR,
    ENGLISH_STOP_WORDS,
    CB_MAX_FEATURES,
    CB_TOP_K,
    CB_CHUNK_SIZE,
    TFIDF_MATRIX_FILE,
    CB_SIMILARITY_FILE,
    TFIDF_VOCAB_FILE,
)

# =============================================================================
#  Tokenization & Vocabulary
# =============================================================================

def _tokenize(text):
    """
    Tách text thành các token, bỏ stop words, lowercase.

    Tham số
    ----------
    text : str

    Trả về
    -------
    list of str
    """
    # Lowercase + bỏ ký tự đặc biệt, chỉ giữ chữ cái và số
    text   = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]


def _generate_ngrams(tokens, n):
    """
    Tạo n-grams từ list tokens.

    Tham số
    ----------
    tokens : list of str
    n : int (1 = unigram, 2 = bigram)

    Trả về
    -------
    list of str (các n-gram nối bằng "_")
    """
    if n == 1:
        return tokens
    return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def build_vocabulary(documents, max_features=10000):
    """
    Xây dựng vocabulary từ các documents.
    Dùng unigram + bigram, chọn top max_features terms theo document frequency.

    Tham số
    ----------
    documents : list of str
    max_features : int
        Số lượng terms tối đa trong vocabulary.

    Trả về
    -------
    dict: {term: index} — ánh xạ term → cột trong ma trận TF
    """
    print("  [Vocab] Đang đếm document frequency...")

    # Bước 1: Với mỗi doc, tạo tập term unique (unigram + bigram)
    df_counter = Counter()

    for doc in documents:
        # Tách doc thành các token
        tokens = _tokenize(doc)
        # (Unigram): Nghĩa là lấy từng từ đơn lẻ một.
        unigrams = _generate_ngrams(tokens, 1)
        # (Bigram): Nghĩa là lấy các cụm từ liền kề nhau.
        bigrams = _generate_ngrams(tokens, 2)
        # Dùng set để đếm document frequency (mỗi term chỉ tính 1 lần/doc)
        unique_terms = set(unigrams + bigrams)
        for term in unique_terms:
            df_counter[term] += 1

    print(f"  [Vocab] Tổng terms unique (trước khi lọc): {len(df_counter):,}")

    # Bước 2: Chọn top max_features theo DF
    top_terms = [term for term, _ in df_counter.most_common(max_features)]
    vocab     = {term: idx for idx, term in enumerate(top_terms)}

    del df_counter, top_terms; gc.collect()

    print(f"  [Vocab] Kích thước vocabulary: {len(vocab):,}")
    return vocab


# =============================================================================
#  TF-IDF Computation
# =============================================================================

def compute_tfidf(documents, vocab, n_docs):
    """
    Tính TF-IDF matrix từ documents và vocabulary.

    TF: sublinear = 1 + log(tf) nếu tf > 0, else 0
    IDF: idf(t) = log((1 + N) / (1 + df(t))) + 1  (smooth idf)
    Chuẩn hóa: L2 normalize mỗi dòng

    Tham số
    ----------
    documents : list of str
    vocab : dict {term: index}
    n_docs : int
        Tổng số documents.

    Trả về
    -------
    csr_matrix: (n_docs × vocab_size), đã L2-normalize
    df_array: numpy array — DF của mỗi term
    """
    vocab_size = len(vocab)
    print(f"  [TF-IDF] Đang xây dựng ma trận TF ({n_docs} docs × {vocab_size} terms)...")

    # Bước 1: Đếm DF cho tất cả terms trong vocab

    # Mảng chứa số lượng sản phẩm có chứa từ phổ biến nhất (DF) cho mỗi term trong vocab
    df_array = np.zeros(vocab_size, dtype=np.float64)

    # Dùng LIL để xây dựng từng dòng
    tf_lil = lil_matrix((n_docs, vocab_size), dtype=np.float64)

    for doc_idx, doc in enumerate(documents):
        tokens = _tokenize(doc)
        unigrams = _generate_ngrams(tokens, 1)
        bigrams  = _generate_ngrams(tokens, 2)
        all_terms = unigrams + bigrams

        # Đếm term frequency trong document này
        tf_counter = Counter()
        for term in all_terms:
            if term in vocab:
                tf_counter[term] += 1

        # Sublinear TF + cập nhật DF
        unique_in_doc = set()
        for term, tf_val in tf_counter.items():
            col = vocab[term]
            sublinear_tf = 1.0 + np.log(tf_val)  # sublinear tf scaling
            tf_lil[doc_idx, col] = sublinear_tf
            unique_in_doc.add(col)

        # Cập nhật DF (mỗi term chỉ tính 1 lần/doc)
        for col in unique_in_doc:
            df_array[col] += 1

    print(f"  [TF-IDF] Đang tính IDF...")

    # Bước 2: Tính smooth IDF
    # idf = log((1 + N) / (1 + df)) + 1
    # idf tỉ lệ nghịch với số lần xuất hiện
    idf = np.log((1.0 + n_docs) / (1.0 + df_array)) + 1.0

    # Bước 3: Nhân TF × IDF
    tfidf_lil = tf_lil.copy()
    for doc_idx in range(n_docs):
        row = tfidf_lil[doc_idx].tocsr()
        if row.nnz == 0:
            continue
        for j in range(row.nnz):
            col = row.indices[j]
            tfidf_lil[doc_idx, col] = row.data[j] * idf[col]

    del tf_lil; gc.collect()

    tfidf_csr = tfidf_lil.tocsr()
    print(f"  [TF-IDF] TF-IDF matrix shape: {tfidf_csr.shape}")
    print(f"  [TF-IDF] Non-zero entries: {tfidf_csr.nnz:,}")

    # Bước 4: L2 normalize mỗi dòng
    print(f"  [TF-IDF] Đang L2 normalize...")
    tfidf_csr = _l2_normalize_rows(tfidf_csr)

    return tfidf_csr, df_array


# =============================================================================
#  Normalization & Similarity
# =============================================================================

def _l2_normalize_rows(sparse_matrix):
    """
    L2 normalize từng dòng của sparse CSR matrix.

    Tham số
    ----------
    sparse_matrix : csr_matrix

    Trả về
    -------
    csr_matrix (có thể sparse hơn nếu có dòng zero)
    """
    n = sparse_matrix.shape[0]
    normalized = lil_matrix(sparse_matrix.shape, dtype=np.float32)

    for i in range(n):
        row = sparse_matrix[i]
        if row.nnz == 0:
            continue

        # Tính L2 norm
        l2_norm = np.sqrt(np.sum(row.data ** 2))
        if l2_norm > 0:
            normalized[i, row.indices] = row.data / l2_norm

    return normalized.tocsr()


# =============================================================================
#  Document Building
# =============================================================================

def build_documents(products_df):
    """
    Tạo text documents cho TF-IDF từ metadata sản phẩm.

    Mỗi document = product_name + " " + department.
    Data loader đã xử lý missing values, không cần check ở đây.

    Tham số
    ----------
    products_df : pd.DataFrame
        Với các cột: product_id, product_name, department

    Trả về
    -------
    tuple: (documents, product_ids)
        documents: list các string
        product_ids: numpy array thẳng hàng với documents
    """
    documents   = []
    product_ids = []

    for _, row in products_df.iterrows():
        doc = f"{row['product_name']} {row['department']}"
        documents.append(doc)
        product_ids.append(row["product_id"])

    return documents, np.array(product_ids)


def build_similarity(tfidf_matrix, top_k=CB_TOP_K, chunk_size=CB_CHUNK_SIZE):
    """
    Tính cosine similarity = chunked dot product giữa các dòng đã L2-normalize.

    Mỗi dòng đã được chuẩn hóa L2 (||row|| = 1). Khi đó mẫu số của công thức Cosine Similarity
    bằng 1, nên Cosine Similarity(row_i, row_j) = dot(row_i, row_j).

    Dùng chunked computation: chunk @ tfidf_matrix.T → chỉ giữ top-K mỗi dòng.

    Tham số
    ----------
    tfidf_matrix : csr_matrix (n_docs × vocab_size), đã L2-normalize
    top_k : int
        Số lượng similar items tối đa cần giữ mỗi dòng.
    chunk_size : int
        Kích thước chunk để tính toán (tránh tràn RAM).

    Trả về
    -------
    csr_matrix (n_docs × n_docs), sparse, chỉ chứa top-K mỗi dòng
    """
    n = tfidf_matrix.shape[0]
    print(f"  [Similarity] Đang tính cosine similarity ({n} × {n}), chunk_size={chunk_size}...")

    # Dùng LIL để xây dựng output
    sim_lil = lil_matrix((n, n), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Lấy chunk dòng
        # Chuyển chunk sang dense để nhân ma trận
        chunk_csr = tfidf_matrix[start:end]

        # Chứa toàn bộ điểm số Cosine Similarity của cụm hiện tại
        sim_chunk = chunk_csr @ tfidf_matrix.T

        # Với mỗi dòng trong chunk, lấy top-K
        for i_local, row_idx in enumerate(range(start, end)):
            row = sim_chunk[i_local]
            if row.nnz == 0:
                continue

            # row_data là giá trị tính toán
            row_data    = row.data
            # row_indices là sản phẩm tương ứng
            row_indices = row.indices

            # Bỏ self-similarity
            mask        = row_indices != row_idx
            row_data    = row_data[mask]
            row_indices = row_indices[mask]

            if len(row_data) == 0:
                continue

            # Lấy top-K
            if len(row_data) <= top_k:
                top_indices = np.arange(len(row_data))
            else:
                top_indices = np.argpartition(row_data, -top_k)[-top_k:]
                # Sắp xếp giảm dần
                top_indices = top_indices[np.argsort(row_data[top_indices])[::-1]]

            sim_lil[row_idx, row_indices[top_indices]] = row_data[top_indices].astype(np.float32)

        del sim_chunk, chunk_csr; gc.collect()

        pct = 100 * end / n
        print(f"    ... {pct:.0f}% ({end:,}/{n:,})")

    sim_csr = sim_lil.tocsr()

    del sim_lil; gc.collect()

    print(f"  [Similarity] Shape: {sim_csr.shape}")
    print(f"  [Similarity] Non-zero entries: {sim_csr.nnz:,}")
    print(f"  [Similarity] Độ sparsity: {100 * sim_csr.nnz / (n ** 2):.2f}%")

    return sim_csr


# =============================================================================
#  Pipeline chính
# =============================================================================

def build_cb_model(products_df, max_features=CB_MAX_FEATURES, top_k=CB_TOP_K):
    """
    Pipeline đầy đủ: tạo documents → vocabulary → TF-IDF → cosine similarity → lưu.

    Tham số
    ----------
    products_df : pd.DataFrame
        Dữ liệu sản phẩm từ data_loader.load_products().
    max_features : int
        Kích thước vocabulary TF-IDF.
    top_k : int
        Giữ top-K similar items mỗi sản phẩm (giảm dung lượng lưu trữ).

    Trả về
    -------
    tuple: (tfidf_matrix, similarity_matrix, vocab)
    """
    print("=" * 50)
    print("Xây dựng Content-Based (CB) Model")
    print("=" * 50)

    # Bước 1: Tạo documents
    print("\n[1/3] Đang tạo text documents...")
    documents, product_ids = build_documents(products_df)
    n_docs = len(documents)
    print(f"  Đã tạo {n_docs} documents")

    # Bước 2: Xây dựng vocabulary + TF-IDF
    print("\n[2/3] TF-IDF vectorization...")
    vocab         = build_vocabulary(documents, max_features=max_features)
    tfidf_matrix, df_array = compute_tfidf(documents, vocab, n_docs)

    # In một vài term top IDF
    idf         = np.log((1.0 + n_docs) / (1.0 + df_array)) + 1.0
    idx_to_term = {v: k for k, v in vocab.items()}

    # Sắp xếp chỉ số từ thấp đến cao theo điểm IDF
    sorted_idx  = np.argsort(idf)

    print(f"   [KIỂM TRA DỮ LIỆU] IN CÁC GIÁ TRỊ IDF KHÁC NHAU ĐỂ SO SÁNH:")

    # 1. In Top 5 từ ĐẠI TRÀ NHẤT (IDF thấp nhất)
    print(f"\n🔹 TOP 5 TỪ ĐẠI TRÀ NHẤT (Xuất hiện quá nhiều, điểm IDF thấp nhất):")
    top_frequent_idx = sorted_idx[:5]  # Lấy 5 thằng đầu tiên
    for idx in top_frequent_idx:
        print(f"    {idx_to_term[idx]:30s}  DF={df_array[idx]:6.0f}  IDF={idf[idx]:.2f}  (Rất loãng)")

    # 2. In Top 5 từ TẦM TRUNG (Nằm ở giữa danh sách)
    print(f"\n🔹 TOP 5 TỪ TẦM TRUNG (Từ khóa vừa phải, tỷ lệ gợi ý lý tưởng):")
    mid_start   = len(sorted_idx) // 2
    top_mid_idx = sorted_idx[mid_start : mid_start + 5]  # Lấy 5 thằng ở giữa
    for idx in top_mid_idx:
        print(f"    {idx_to_term[idx]:30s}  DF={df_array[idx]:6.0f}  IDF={idf[idx]:.2f}  (Lý tưởng)")

    # 3. In Top 5 từ ĐỘC LẠ NHẤT (IDF cao nhất)
    print(f"\n🔹 TOP 5 TỪ ĐỘC LẠ NHẤT (Xuất hiện cực ít, điểm IDF cao nhất):")
    top_rare_idx = sorted_idx[::-1][:5]  # Đảo ngược lấy 5 thằng cuối
    for idx in top_rare_idx:
        print(f"    {idx_to_term[idx]:30s}  DF={df_array[idx]:6.0f}  IDF={idf[idx]:.2f}  (Hàng hiếm)")
    print(f"\n" + "=" * 70)

    del documents, df_array, idf; gc.collect()

    # Bước 3: Cosine similarity
    print("\n[3/3] Đang tính cosine similarity...")
    sim_matrix = build_similarity(tfidf_matrix, top_k=top_k, chunk_size=CB_CHUNK_SIZE)

    return tfidf_matrix, sim_matrix, vocab


# =============================================================================
#  Lưu model
# =============================================================================

def save_model(tfidf_matrix, sim_matrix, vocab):
    """
    Lưu CB model outputs vào thư mục models/.

    Tham số
    ----------
    tfidf_matrix : csr_matrix
    sim_matrix : csr_matrix
    vocab : dict {term: index}
    """
    print("\nĐang lưu CB model outputs...")

    # Lưu sparse matrices
    save_npz(MODELS_DIR / TFIDF_MATRIX_FILE, tfidf_matrix)
    print(f"  Đã lưu: models/{TFIDF_MATRIX_FILE}")

    save_npz(MODELS_DIR / CB_SIMILARITY_FILE, sim_matrix)
    print(f"  Đã lưu: models/{CB_SIMILARITY_FILE}")

    # Lưu vocabulary (thay vì pickle vectorizer)
    with open(MODELS_DIR / TFIDF_VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  Đã lưu: models/{TFIDF_VOCAB_FILE}")

    print("\nCB model hoàn tất!")


if __name__ == "__main__":
    from src.utils.data_loader import load_products

    # Tải dữ liệu
    products_df = load_products()

    # Xây dựng model
    tfidf_matrix, sim_matrix, vocab = build_cb_model(
        products_df, max_features=CB_MAX_FEATURES, top_k=CB_TOP_K
    )

    # Lưu
    save_model(tfidf_matrix, sim_matrix, vocab)