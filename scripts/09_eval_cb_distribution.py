"""
09 — Đánh giá CB similarity trên thang 0-1 (step 0.25) kèm word overlap.
Không phân tích threshold, không vẽ biểu đồ — chỉ đánh giá chất lượng 0-1.

Yêu cầu đã chạy:
    1. scripts/01_load_data.py    → data/processed/products.parquet
    2. scripts/02_cb_filter.py    → models/cb_filter/product_vectors.npz + product_id_to_idx.json

Output:
    results/cb_similarity_distribution/cb_similarity_samples.csv
"""
import json
import os
import sys
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import MODEL_DIR, PROCESSED_DIR, RESULT_DIR
from src.features.vectorizer import cb_similarity

RANDOM_SEED = 42
N_SAMPLE_PAIRS = 200_000
N_EXAMPLES_PER_BUCKET = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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


def sample_pairs(product_id_to_idx, n_pairs=N_SAMPLE_PAIRS):
    all_ids = list(product_id_to_idx.keys())
    pairs = []
    for _ in range(n_pairs):
        a = random.choice(all_ids)
        b = random.choice(all_ids)
        if a != b:
            pairs.append((a, b))
    return pairs


def compute_similarities_and_overlaps(product_vectors, product_id_to_idx, pairs, products):
    pid_to_name = dict(zip(products['product_id'], products['product_name']))

    sims, overlaps, jaccards, names_a, names_b = [], [], [], [], []

    n_total = len(pairs)
    for i, (a, b) in enumerate(pairs):
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{n_total}")

        if a not in product_id_to_idx or b not in product_id_to_idx:
            continue

        idx_a = product_id_to_idx[a]
        idx_b = product_id_to_idx[b]
        sim = cb_similarity(product_vectors, idx_a, [idx_b])[0]

        name_a = pid_to_name.get(a, '')
        name_b = pid_to_name.get(b, '')
        tokens_a = tokenize(name_a)
        tokens_b = tokenize(name_b)

        overlap = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jac = overlap / union if union > 0 else 0

        sims.append(sim)
        overlaps.append(overlap)
        jaccards.append(jac)
        names_a.append(name_a)
        names_b.append(name_b)

    return {
        'similarity': np.array(sims),
        'word_overlap': np.array(overlaps),
        'jaccard': np.array(jaccards),
        'name_a': names_a,
        'name_b': names_b,
    }


def plot_histogram(similarities, save_path):
    """Vẽ 1 histogram đơn giản: y=số cặp, x=cosine similarity (0-1)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(similarities, bins=40, range=(0, 1), color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Số cặp')
    ax.set_title(f'Phân bố CB Similarity (n={len(similarities):,})')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Histogram saved: {save_path}")


def print_bucket_table(data):
    """In bảng 4 bucket 0-1 step 0.25 kèm word overlap + ví dụ."""
    sims = data['similarity']
    overlaps = data['word_overlap']
    jaccards = data['jaccard']
    names_a = data['name_a']
    names_b = data['name_b']

    buckets = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00)]

    print()
    print("=" * 120)
    print("  ĐÁNH GIÁ CB SIMILARITY TRÊN THANG 0-1 (step 0.25)")
    print("=" * 120)

    header = (
        f"{'Bucket':<18} | {'n pairs':>8} | {'%':>5} | "
        f"{'avg overlap':>10} | {'avg jaccard':>10} | {'ví dụ'}".format()
    )
    print(header)
    print("-" * 120)

    for lo, hi in buckets:
        mask = (sims >= lo) & (sims < hi)
        n = mask.sum()
        pct = n / len(sims) * 100 if len(sims) > 0 else 0

        avg_overlap = overlaps[mask].mean() if n > 0 else 0
        avg_jac = jaccards[mask].mean() if n > 0 else 0

        # Lấy 5 ví dụ (ưu tiên similarity cao trong bucket)
        indices = np.where(mask)[0]
        if n > 0:
            sorted_idx = indices[np.argsort(sims[indices])[::-1]]
            sample_idx = sorted_idx[:N_EXAMPLES_PER_BUCKET]
        else:
            sample_idx = []

        bucket_label = f"[{lo:.2f}, {hi:.2f})"
        print(f"{bucket_label:<18} | {n:>8} | {pct:>4.1f}% | "
              f"{avg_overlap:>10.2f} | {avg_jac:>10.4f}")

        for idx in sample_idx:
            a_name = names_a[idx] if idx < len(names_a) else '?'
            b_name = names_b[idx] if idx < len(names_b) else '?'
            overlap_val = overlaps[idx] if idx < len(overlaps) else 0
            sim_val = sims[idx] if idx < len(sims) else 0

            tokens_a = tokenize(a_name)
            tokens_b = tokenize(b_name)
            common = tokens_a & tokens_b
            common_str = ", ".join(sorted(common)) if common else "(không có)"

            print(f"  {'':>18} | {'':>8} | {'':>5} | "
                  f"{'':>10} | {'':>10} | "
                  f"sim={sim_val:.3f} | overlap={overlap_val} | "
                  f"\"{a_name}\" ~ \"{b_name}\" | trùng: {common_str}")
        print("-" * 120)

    print(f"\nTổng số cặp hợp lệ: {len(sims)}")
    print("=" * 120)


def save_raw_data(data):
    save_dir = os.path.join(RESULT_DIR, "cb_similarity_distribution")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame({
        'similarity': data['similarity'],
        'word_overlap': data['word_overlap'],
        'jaccard': data['jaccard'],
        'product_a': data['name_a'],
        'product_b': data['name_b'],
    })
    path = os.path.join(save_dir, "cb_similarity_samples.csv")
    df.to_csv(path, index=False)
    print(f"\nRaw data saved: {path}")


def main():
    print("=" * 60)
    print("  ĐÁNH GIÁ CB SIMILARITY (0-1) + WORD OVERLAP")
    print("=" * 60)

    print("\n1. Loading data...")
    products, product_vectors, product_id_to_idx = load_data()

    print(f"\n2. Sampling {N_SAMPLE_PAIRS:,} pairs...")
    pairs = sample_pairs(product_id_to_idx, N_SAMPLE_PAIRS)
    print(f"   Sampled: {len(pairs)} pairs")

    print(f"\n3. Computing similarities + word overlap...")
    data = compute_similarities_and_overlaps(
        product_vectors, product_id_to_idx, pairs, products
    )
    print(f"   Computed: {len(data['similarity'])} pairs")

    print_bucket_table(data)

    save_dir = os.path.join(RESULT_DIR, "cb_similarity_distribution")
    os.makedirs(save_dir, exist_ok=True)
    plot_histogram(data['similarity'], os.path.join(save_dir, "histogram.png"))

    save_raw_data(data)

    print("\n  HOÀN TẤT!")


if __name__ == '__main__':
    main()