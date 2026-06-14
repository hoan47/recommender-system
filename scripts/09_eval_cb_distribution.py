"""
09 — Đánh giá CB similarity theo số từ trùng nhau.
So sánh 3 loại similarity: TF-IDF, Count Vectorizer (L2-norm), Ensemble (alpha * Count + (1-alpha) * TF-IDF).
Với mỗi overlap = 1..10, tìm 5 cặp ngẫu nhiên có đúng N từ chung, tính similarity.

Không vẽ biểu đồ, không phân tích bucket.
"""
import json
import os
import sys
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse

from src.config import MODEL_DIR, PROCESSED_DIR, RESULT_DIR, CB_ALPHA
from src.features.vectorizer import cb_similarity

N_EXAMPLES = 20         # cặp cho mỗi overlap
MAX_OVERLAP = 10       # từ 1 đến 10 từ trùng
MAX_TRIES = 10_000_000  # giới hạn lần thử tránh treo


def load_data():
    products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))

    # TF-IDF vectors
    tfidf_vectors = scipy.sparse.load_npz(
        os.path.join(MODEL_DIR, "cb_filter", "tfidf_vectors.npz")
    )
    # Count vectors (L2-normalized)
    count_vectors = scipy.sparse.load_npz(
        os.path.join(MODEL_DIR, "cb_filter", "count_vectors.npz")
    )

    with open(os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")) as f:
        product_id_to_idx = {int(k): v for k, v in json.load(f).items()}

    return products, tfidf_vectors, count_vectors, product_id_to_idx


def tokenize(name):
    if not name or not isinstance(name, str):
        return set()
    return set(name.lower().split())


def _overlap_similarity(count_vectors, idx_a, idx_b_list):
    """Tính Overlap Score giữa product_a (idx_a) và từng candidate trong idx_b_list."""
    vec_a = count_vectors[idx_a]
    vecs_b = count_vectors[idx_b_list]

    # Nhị phân hóa
    bin_a = vec_a.astype(bool).astype(np.float64)
    bin_b = vecs_b.astype(bool).astype(np.float64)

    intersection = bin_b.dot(bin_a.T).toarray().ravel()
    sum_a = bin_a.sum()
    sum_b = np.array(bin_b.sum(axis=1)).ravel()
    denom = np.minimum(sum_a, sum_b)

    return np.divide(intersection, denom,
                     out=np.zeros_like(intersection, dtype=float),
                     where=denom != 0)


def find_pairs_by_overlap(products, tfidf_vectors, count_vectors, product_id_to_idx):
    """
    Duyệt random pairs, nhóm theo số từ trùng.
    Trả về dict: overlap -> list of (sim_tfidf, sim_count, sim_ensemble, name_a, name_b, common_tokens)
    """
    alpha = CB_ALPHA
    metric = 'overlap'
    pid_to_name = dict(zip(products['product_id'], products['product_name']))
    all_ids = list(product_id_to_idx.keys())

    result = {n: [] for n in range(1, MAX_OVERLAP + 1)}
    needed = set(range(1, MAX_OVERLAP + 1))

    tried = 0
    while needed and tried < MAX_TRIES:
        tried += 1
        if tried % 50000 == 0:
            print(f"  ... đã thử {tried:,} cặp, còn thiếu {len(needed)} overlap: {sorted(needed)}")

        a = random.choice(all_ids)
        b = random.choice(all_ids)
        if a == b:
            continue
        if a not in product_id_to_idx or b not in product_id_to_idx:
            continue

        name_a = pid_to_name.get(a, '')
        name_b = pid_to_name.get(b, '')
        tokens_a = tokenize(name_a)
        tokens_b = tokenize(name_b)
        overlap = len(tokens_a & tokens_b)

        if overlap not in needed:
            continue
        if len(result[overlap]) >= N_EXAMPLES:
            if overlap in needed:
                needed.discard(overlap)
            continue

        # Tính similarity trên cả 2 vectorizers
        idx_a = product_id_to_idx[a]
        idx_b = product_id_to_idx[b]

        sim_tfidf = cb_similarity(tfidf_vectors, idx_a, [idx_b])[0]
        if metric == 'overlap':
            sim_count = _overlap_similarity(count_vectors, idx_a, [idx_b])[0]
        else:
            sim_count = cb_similarity(count_vectors, idx_a, [idx_b])[0]
        sim_ensemble = alpha * sim_count + (1.0 - alpha) * sim_tfidf

        common = tokens_a & tokens_b
        result[overlap].append((sim_tfidf, sim_count, sim_ensemble, name_a, name_b, sorted(common)))

        if len(result[overlap]) >= N_EXAMPLES:
            needed.discard(overlap)

    return result


def print_results(data):
    print()
    print("=" * 140)
    print("  CB SIMILARITY THEO SỐ TỪ TRÙNG (TF-IDF | Count | Ensemble)")
    print("=" * 140)

    for overlap in range(1, MAX_OVERLAP + 1):
        pairs = data.get(overlap, [])
        print(f"\n--- overlap = {overlap} ({len(pairs)} cặp) ---")
        print(f"  {'TF-IDF':>8} | {'Count':>8} | {'Ensemble':>8} | Sản phẩm A ~ Sản phẩm B | Từ trùng")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+{'-'*45}+-{'-'*30}")
        for sim_t, sim_c, sim_e, name_a, name_b, common in pairs:
            common_str = ", ".join(common) if common else "(không có)"
            print(f"  {sim_t:>8.4f} | {sim_c:>8.4f} | {sim_e:>8.4f} | \"{name_a}\" ~ \"{name_b}\" | {common_str}")

    print("\n" + "=" * 140)


def main():
    print("=" * 60)
    print("  ĐÁNH GIÁ CB SIMILARITY THEO WORD OVERLAP")
    print(f"  Ensemble: alpha(Count)={CB_ALPHA}")
    print("=" * 60)

    print("\n1. Loading data...")
    products, tfidf_vectors, count_vectors, product_id_to_idx = load_data()
    print(f"   products={len(products)}")
    print(f"   TF-IDF vectors={tfidf_vectors.shape}")
    print(f"   Count vectors={count_vectors.shape}")

    print(f"\n2. Tìm cặp cho overlap 1->{MAX_OVERLAP}, mỗi cái {N_EXAMPLES} cặp...")
    data = find_pairs_by_overlap(products, tfidf_vectors, count_vectors, product_id_to_idx)

    print_results(data)

    # Lưu CSV
    rows = []
    for overlap in range(1, MAX_OVERLAP + 1):
        for sim_t, sim_c, sim_e, name_a, name_b, common in data.get(overlap, []):
            rows.append({
                'overlap': overlap,
                'sim_tfidf': sim_t,
                'sim_count': sim_c,
                'sim_ensemble': sim_e,
                'product_a': name_a,
                'product_b': name_b,
                'common_tokens': ", ".join(common),
            })
    df = pd.DataFrame(rows)
    save_dir = os.path.join(RESULT_DIR, "cb_similarity_distribution")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "cb_overlap_samples.csv")
    df.to_csv(path, index=False)
    print(f"\nRaw data saved: {path}")

    print("\n  HOÀN TẤT!")


if __name__ == '__main__':
    main()