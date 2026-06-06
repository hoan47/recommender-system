"""
Content-Based vectors + similarity (phong cách dict-based, gọn nhẹ)

CB dùng để LỌC sản phẩm tương tự (substitute) ra khỏi gợi ý.
KHÔNG phải model gợi ý chính. Output là dict{pid: dict{vocab_idx: tfidf_val}}.
Khi cần tính similarity giữa 2 sản phẩm, chỉ cần dot product giữa 2 dict
(key chung nhau). Không cần ma trận sparse 50Kx50K tốn RAM.
"""

import re
import math
import gc
import json
from collections import Counter
from tqdm import tqdm

from src.config import MODELS_DIR, CB_MIN_DF, CB_MAX_DF, CB_MAX_FEATURES

# File lưu CB vectors dạng json (vì dict-based, không phải ma trận)
CB_FILE = MODELS_DIR / "cb_vectors.json"
# Biến toàn cục lưu vectors: {pid: {vocab_idx: tfidf_val}}
# vocab_idx là index của term trong vocabulary (0..CB_MAX_FEATURES-1)
# tfidf_val là giá trị TF-IDF đã L2-normalize
cb_vectors = {}

def build_cb_vectors(products_df, prior_df):
    """
    Xây dựng CB vectors từ tên sản phẩm (content-based).
    Chỉ xây cho sản phẩm có xuất hiện trong prior (frequent items).
    Các sản phẩm long-tail không có trong prior sẽ không được index.

    Quy trình:
        1. Tokenize tên sản phẩm → unigram + bigram
        2. Đếm Document Frequency (DF) cho mỗi term
        3. Build vocabulary: lọc term theo min_df/max_df, giữ tối đa max_features
        4. Tính IDF smooth: idf = log((1+N)/(1+df)) + 1
        5. TF sublinear: tf = 1 + log(count)
        6. TF-IDF = tf * idf, sau đó L2-normalize toàn bộ vector
    """
    global cb_vectors
    print("\n  [CB] Building content-based vectors ...")

    # Chỉ lấy sản phẩm có xuất hiện trong prior
    prod_info = products_df.set_index('product_id')
    freq_pids = set(prior_df['product_id'].unique())
    pids = [pid for pid in freq_pids if pid in prod_info.index]

    # Bước 1: Tokenize mỗi sản phẩm, sinh unigram + bigram
    # Lưu toàn bộ tokens để dùng lại khi tính TF-IDF
    doc_tokens = {}
    doc_freq = Counter()
    for pid in tqdm(pids, desc="  Tokenizing"):
        raw = str(prod_info.loc[pid, 'product_name'])
        # Lowercase, chỉ giữ chữ cái và số (bỏ ký tự đặc biệt)
        text = re.sub(r'[^a-z0-9 ]', ' ', raw.lower()).strip()
        tokens = text.split()
        ngrams = list(tokens)  # unigram: từng từ đơn
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]} {tokens[i+1]}")  # bigram: cặp từ liền kề
        doc_tokens[pid] = ngrams
        # Đếm document frequency (mỗi term chỉ tính 1 lần/sản phẩm)
        for t in set(ngrams):
            doc_freq[t] += 1

    # Bước 2: Build vocabulary
    # Sắp xếp term theo DF giảm dần (term phổ biến nhất trước)
    # Chỉ giữ term có DF >= min_df VÀ DF/tổng <= max_df
    n_docs = len(pids)
    vocab = {}
    idx = 0
    for t, df in sorted(doc_freq.items(), key=lambda x: -x[1]):
        if df >= CB_MIN_DF and (df / n_docs) <= CB_MAX_DF:
            vocab[t] = idx
            idx += 1
            if idx >= CB_MAX_FEATURES:
                break

    # Bước 3: Tính IDF cho mỗi term (công thức smooth: +1 ở cả tử và mẫu)
    idf = {t: math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0 for t in vocab}

    # Bước 4: TF-IDF + L2 normalize cho từng sản phẩm
    for pid in tqdm(pids, desc="  TF-IDF"):
        tokens = doc_tokens[pid]
        cnt = Counter(tokens)  # term frequency trong sản phẩm này
        vec = {}
        norm_sq = 0.0
        for t, tf in cnt.items():
            if t not in vocab:
                continue
            # TF sublinear: 1 + log(tf) — giảm ảnh hưởng của term xuất hiện nhiều lần
            val = (1.0 + math.log(tf)) * idf[t]
            vec[vocab[t]] = val
            norm_sq += val * val  # accum để tính L2 norm
        # L2 normalize: chia mỗi giá trị cho sqrt(tổng bình phương)
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for k in vec:
                vec[k] /= norm
        cb_vectors[pid] = vec

    # Dọn dẹp bộ nhớ
    del doc_tokens, doc_freq, vocab, idf, prod_info
    gc.collect()
    print(f"  [CB] Done: {len(cb_vectors):,} vectors")

def cb_similarity(pid_a, pid_b):
    """
    Cosine similarity giữa 2 sản phẩm dùng dict-based dot product.
    Vì cả 2 vector đã L2-normalize, cosine similarity = dot product.
    Chỉ nhân các key chung nhau (term xuất hiện ở cả 2 sản phẩm).
    """
    va = cb_vectors.get(pid_a)
    vb = cb_vectors.get(pid_b)
    if not va or not vb:
        return 0.0  # Cold-start: không có vector → similarity = 0
    # Lấy vector ngắn hơn làm vòng lặp ngoài (tối ưu tốc độ)
    if len(va) > len(vb):
        va, vb = vb, va
    return sum(va[t] * vb[t] for t in va if t in vb)

def save():
    """Lưu cb_vectors ra file json (key là str vì json không hỗ trợ int key)"""
    out = {}
    for pid, vec in cb_vectors.items():
        out[str(pid)] = {str(k): float(v) for k, v in vec.items()}
    with open(CB_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  [CB] Saved: {CB_FILE}")

def load():
    """Load cb_vectors từ file json, chuyển key từ str về int"""
    global cb_vectors
    with open(CB_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cb_vectors = {int(pid): {int(k): float(v) for k, v in vec.items()}
                  for pid, vec in raw.items()}
    print(f"  [CB] Loaded: {len(cb_vectors):,} vectors")

if __name__ == "__main__":
    from src.data_loader import load_products, load_prior
    products_df = load_products()
    prior_df = load_prior()
    build_cb_vectors(products_df, prior_df)
    save()
    # Demo: tính similarity giữa 2 sản phẩm bất kỳ
    print("  [CB] Demo: pid1=1, pid2=3, similarity=", cb_similarity(1, 3))