"""
09 — Đánh giá CB similarity theo số từ trùng nhau.
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

from src.config import MODEL_DIR, PROCESSED_DIR, RESULT_DIR
from src.features.vectorizer import cb_similarity

N_EXAMPLES = 20         # cặp cho mỗi overlap
MAX_OVERLAP = 10       # từ 1 đến 10 từ trùng
MAX_TRIES = 5_000_000  # giới hạn lần thử tránh treo

def load_data():
    products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
    product_vectors = scipy.sparse.load_npz(
        os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")
    )
    with open(os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")) as f:
        product_id_to_idx = {int(k): v for k, v in json.load(f).items()}
    return products, product_vectors, product_id_to_idx


def tokenize(name):
    if not name or not isinstance(name, str):
        return set()
    return set(name.lower().split())


def find_pairs_by_overlap(products, product_vectors, product_id_to_idx):
    """
    Duyệt random pairs, nhóm theo số từ trùng.
    Trả về dict: overlap -> list of (sim, name_a, name_b, common_tokens)
    """
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

        # Tính similarity
        idx_a = product_id_to_idx[a]
        idx_b = product_id_to_idx[b]
        sim = cb_similarity(product_vectors, idx_a, [idx_b])[0]

        common = tokens_a & tokens_b
        result[overlap].append((sim, name_a, name_b, sorted(common)))

        if len(result[overlap]) >= N_EXAMPLES:
            needed.discard(overlap)

    return result


def print_results(data):
    print()
    print("=" * 120)
    print("  CB SIMILARITY THEO SỐ TỪ TRÙNG")
    print("=" * 120)

    for overlap in range(1, MAX_OVERLAP + 1):
        pairs = data.get(overlap, [])
        print(f"\n--- overlap = {overlap} ({len(pairs)} cặp) ---")
        for sim, name_a, name_b, common in pairs:
            common_str = ", ".join(common) if common else "(không có)"
            print(f"  sim={sim:.4f} | \"{name_a}\" ~ \"{name_b}\" | trùng: {common_str}")

    print("\n" + "=" * 120)


def main():
    print("=" * 60)
    print("  ĐÁNH GIÁ CB SIMILARITY THEO WORD OVERLAP")
    print("=" * 60)

    print("\n1. Loading data...")
    products, product_vectors, product_id_to_idx = load_data()
    print(f"   products={len(products)}, vectors={product_vectors.shape}")

    print(f"\n2. Tìm cặp cho overlap 1->{MAX_OVERLAP}, mỗi cái {N_EXAMPLES} cặp...")
    data = find_pairs_by_overlap(products, product_vectors, product_id_to_idx)

    print_results(data)

    # Lưu CSV
    rows = []
    for overlap in range(1, MAX_OVERLAP + 1):
        for sim, name_a, name_b, common in data.get(overlap, []):
            rows.append({
                'overlap': overlap,
                'similarity': sim,
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