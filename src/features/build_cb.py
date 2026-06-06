"""
Content-Based vectors + similarity (phong cách dict-based, gọn nhẹ)

CB dùng để LỌC sản phẩm tương tự (substitute) ra khỏi gợi ý.
KHÔNG phải model gợi ý chính. Output là dict{pid: dict{vocab_idx: tfidf_val}}.
Khi cần tính similarity giữa 2 sản phẩm, chỉ cần dot product giữa 2 dict
(key chung nhau). Không cần ma trận sparse 50Kx50K tốn RAM.

Optimizations:
  1. Vectorize vocabulary building (sort + filter bằng numpy arrays)
  2. Counter batch processing thay vì từng product một (dùng numpy cho IDF)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import re
import math
import gc
import json
from collections import Counter
import numpy as np
from tqdm import tqdm

from src.config import MODELS_DIR, CB_MIN_DF, CB_MAX_DF, CB_MAX_FEATURES

# File lưu CB vectors dạng json (vì dict-based, không phải ma trận)
CB_FILE = MODELS_DIR / "cb_vectors.json"
# Biến toàn cục lưu vectors: {pid: {vocab_idx: tfidf_val}}
cb_vectors = {}


def _tokenize_product(text):
    """
    Tokenize tên sản phẩm: lowercase, chỉ giữ chữ cái/số, unigram + bigram.
    Thuần Python (regex-based), nhanh vì chỉ ~50K sản phẩm, không phải bottleneck.
    """
    text = re.sub(r'[^a-z0-9 ]', ' ', text.lower()).strip()
    tokens = text.split()
    ngrams = list(tokens)  # unigram
    for i in range(len(tokens) - 1):
        ngrams.append(f"{tokens[i]} {tokens[i+1]}")  # bigram
    return ngrams


def build_cb_vectors(products_df, prior_df):
    """
    Xây dựng CB vectors từ tên sản phẩm (content-based).
    Chỉ xây cho sản phẩm có xuất hiện trong prior (frequent items).
    
    Quy trình:
        1. Tokenize tên sản phẩm → unigram + bigram
        2. Đếm Document Frequency (DF) cho mỗi term
        3. Build vocabulary: lọc term bằng numpy arrays, sắp xếp theo DF
        4. Tính IDF smooth: idf = log((1+N)/(1+df)) + 1
        5. TF sublinear: tf = 1 + log(count)
        6. TF-IDF = tf * idf, sau đó L2-normalize
    """
    global cb_vectors
    print("\n  [CB] Building content-based vectors ...")

    # Chỉ lấy sản phẩm có xuất hiện trong prior
    prod_info = products_df.set_index('product_id')
    freq_pids = set(prior_df['product_id'].unique())
    pids = [pid for pid in freq_pids if pid in prod_info.index]

    # Bước 1: Tokenize mỗi sản phẩm
    doc_tokens = {}
    doc_freq = Counter()
    for pid in tqdm(pids, desc="  Tokenizing"):
        raw = str(prod_info.loc[pid, 'product_name'])
        ngrams = _tokenize_product(raw)
        doc_tokens[pid] = ngrams
        for t in set(ngrams):
            doc_freq[t] += 1

    # Bước 2: Build vocabulary dùng numpy filtering
    n_docs = len(pids)
    # Convert counter → arrays để xử lý nhanh
    terms = np.array(list(doc_freq.keys()))
    dfs = np.array([doc_freq[t] for t in terms], dtype=np.int32)
    
    # Filter theo min_df và max_df
    mask = (dfs >= CB_MIN_DF) & (dfs.astype(float) / n_docs <= CB_MAX_DF)
    filtered_terms = terms[mask]
    filtered_dfs = dfs[mask]
    
    # Sắp xếp theo DF giảm dần, giữ max_features
    sort_order = np.argsort(-filtered_dfs)
    n_keep = min(len(filtered_terms), CB_MAX_FEATURES)
    final_terms = filtered_terms[sort_order[:n_keep]]
    
    vocab = {t: i for i, t in enumerate(final_terms)}
    del terms, dfs, mask, filtered_terms, filtered_dfs, sort_order
    gc.collect()

    # Bước 3: Tính IDF cho mỗi term
    idf = {t: math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0 for t in vocab}

    # Bước 4: TF-IDF + L2 normalize cho từng sản phẩm
    for pid in tqdm(pids, desc="  TF-IDF"):
        tokens = doc_tokens[pid]
        cnt = Counter(tokens)
        vec = {}
        norm_sq = 0.0
        for t, tf in cnt.items():
            if t not in vocab:
                continue
            val = (1.0 + math.log(tf)) * idf[t]
            vec[vocab[t]] = val
            norm_sq += val * val
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for k in vec:
                vec[k] /= norm
        cb_vectors[pid] = vec

    del doc_tokens, doc_freq, vocab, idf, prod_info
    gc.collect()
    print(f"  [CB] Done: {len(cb_vectors):,} vectors, "
          f"vocab_size={n_keep:,}")


def cb_similarity(pid_a, pid_b):
    """
    Cosine similarity giữa 2 sản phẩm dùng dict-based dot product.
    """
    va = cb_vectors.get(pid_a)
    vb = cb_vectors.get(pid_b)
    if not va or not vb:
        return 0.0
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
    print("  [CB] Demo: pid1=1, pid2=3, similarity=", cb_similarity(1, 3))