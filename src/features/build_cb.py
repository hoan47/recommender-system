"""
Content-Based vectors + similarity
CB dùng để LỌC sản phẩm tương tự (substitute), KHÔNG phải model gợi ý chính
"""
import re, math, gc, json
from collections import Counter
from tqdm import tqdm

from src.config import MODELS_DIR, CB_MIN_DF, CB_MAX_DF, CB_MAX_FEATURES

CB_FILE = MODELS_DIR / "cb_vectors.json"
cb_vectors = {}  # {pid: {vocab_idx: tfidf_val}}

def build_cb_vectors(products_df, prior_df):
    """
    Xây CB vectors dạng dict từ tên sản phẩm
    Chỉ build cho sản phẩm có trong prior (frequent items)
    """
    global cb_vectors
    print("\n  [CB] Building content-based vectors ...")

    prod_info = products_df.set_index('product_id')
    # Chỉ lấy sản phẩm có trong prior
    freq_pids = set(prior_df['product_id'].unique())
    pids = [pid for pid in freq_pids if pid in prod_info.index]

    # 1. Tokenize + đếm DF
    doc_tokens = {}
    doc_freq = Counter()
    for pid in tqdm(pids, desc="  Tokenizing"):
        raw = str(prod_info.loc[pid, 'product_name'])
        text = re.sub(r'[^a-z0-9 ]', ' ', raw.lower()).strip()
        tokens = text.split()
        ngrams = list(tokens)
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]} {tokens[i+1]}")
        doc_tokens[pid] = ngrams
        for t in set(ngrams):
            doc_freq[t] += 1

    # 2. Build vocab theo DF
    n_docs = len(pids)
    vocab = {}
    idx = 0
    for t, df in sorted(doc_freq.items(), key=lambda x: -x[1]):
        if df >= CB_MIN_DF and (df / n_docs) <= CB_MAX_DF:
            vocab[t] = idx
            idx += 1
            if idx >= CB_MAX_FEATURES:
                break

    # 3. IDF
    idf = {t: math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0 for t in vocab}

    # 4. TF-IDF + L2 normalize
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
    print(f"  [CB] Done: {len(cb_vectors):,} vectors")

def cb_similarity(pid_a, pid_b):
    """Cosine similarity giữa 2 product vectors (dict-based)"""
    va = cb_vectors.get(pid_a)
    vb = cb_vectors.get(pid_b)
    if not va or not vb:
        return 0.0
    if len(va) > len(vb):
        va, vb = vb, va
    return sum(va[t] * vb[t] for t in va if t in vb)

def save():
    """Lưu cb_vectors dạng {str(pid): {str(idx): val}} vì json cần str keys"""
    out = {}
    for pid, vec in cb_vectors.items():
        out[str(pid)] = {str(k): float(v) for k, v in vec.items()}
    with open(CB_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  [CB] Saved: {CB_FILE}")

def load():
    """Load cb_vectors từ json"""
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