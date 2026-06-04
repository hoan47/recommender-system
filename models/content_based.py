import math
import gc
import re
from collections import Counter
from tqdm import tqdm

import data_loader as dl
import dept_direction as dd
from config import CB_MAX_DF, CB_MIN_DF

cb_vectors: dict = {}


def build(prod_dept: dict) -> None:
    global cb_vectors

    prod_info = dl.products.set_index("product_id")
    pids = [pid for pid in dl.frequent_items if pid in prod_info.index]

    doc_tokens: dict = {}
    doc_freq = Counter()

    for pid in tqdm(pids, desc="  [CB] Tokenizing", ncols=80):
        raw = str(prod_info.loc[pid, "product_name_vi"])
        text = re.sub(r"[^a-z0-9 ]", " ", raw.lower()).strip()
        tokens = text.split()
        ngrams = list(tokens)
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]} {tokens[i + 1]}")
        doc_tokens[pid] = ngrams
        for t in set(ngrams):
            doc_freq[t] += 1

    n_docs = len(pids)
    vocab: dict = {}
    idx = 0
    for t, df in sorted(doc_freq.items(), key=lambda x: -x[1]):
        if df >= CB_MIN_DF and (df / n_docs) <= CB_MAX_DF:
            vocab[t] = idx
            idx += 1
            if idx >= 30_000:
                break

    idf = {
        t: math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0
        for t in vocab
    }

    for pid in tqdm(pids, desc="  [CB] TF-IDF", ncols=80):
        tokens = doc_tokens[pid]
        counts = Counter(tokens)
        vec: dict = {}
        norm_sq = 0.0
        for t, tf in counts.items():
            if t not in vocab:
                continue
            val = (1.0 + math.log(tf)) * idf[t]
            vec[vocab[t]] = val
            norm_sq += val * val
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for t_idx in vec:
                vec[t_idx] /= norm
        cb_vectors[pid] = vec

    del doc_tokens, doc_freq, vocab, idf, prod_info
    gc.collect()
    print(f"  [CB] Done — {len(cb_vectors):,} vectors")


def similarity(pid_a: int, pid_b: int) -> float:
    va = cb_vectors.get(pid_a)
    vb = cb_vectors.get(pid_b)
    if not va or not vb:
        return 0.0
    if len(va) > len(vb):
        va, vb = vb, va
    return sum(va[t] * vb[t] for t in va if t in vb)


def recommend(product_id: int, k: int = 100) -> list:
    if product_id not in cb_vectors:
        return []
    scores = {
        pid: similarity(product_id, pid)
        for pid in dl.frequent_items
        if pid != product_id
    }
    scores = {p: s for p, s in scores.items() if s > 0}
    recs = sorted(scores, key=lambda p: -scores[p])
    recs = dd.filter_by_direction(product_id, recs)
    return recs[:k]
