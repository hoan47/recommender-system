"""
09 — Đánh giá phân bố cosine similarity của CB (Content-Based).
Giúp chọn ENS_CB_THRESHOLD phù hợp dựa trên dữ liệu thực tế.

Yêu cầu đã chạy:
    1. scripts/01_load_data.py    → tạo data/processed/products_vi.csv
    2. scripts/02_cb_filter.py    → tạo models/cb_filter/product_vectors.npz + product_id_to_idx.json

⚠️ LƯU Ý QUAN TRỌNG (dễ nhầm):
    - CB vectors được TF-IDF train trên tên sản phẩm TIẾNG VIỆT (product_name_vi)
    - File này load products_vi.csv (tiếng Việt) để word overlap analysis match với CB vectors
    - KHÔNG dùng products.parquet (tiếng Anh) ở đây — sẽ dẫn đến sai lệch word overlap & similarity
    - products.parquet (49,689 records, English) dùng cho collaborative models (Ochiai, Item2Vec, DeepWalk...)
    - products_vi.csv (49,688 records, Vietnamese) dùng cho CB Filter + CB evaluation

Output: results/cb_similarity_distribution/ (ảnh + CSV thống kê)
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

# Seed cho reproducible
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Số cặp cần sample — toàn bộ 49K^2 là quá lớn
N_SAMPLE_PAIRS = 200000
# Top-K hiển thị trong báo cáo
TOP_K_SHOW = 20


def load_data():
    """
    Load products (tiếng Việt) + product vectors (trained trên tiếng Việt).

    Dùng products_vi.csv thay vì products.parquet vì CB vectors được train
    trên product_name_vi (xem 02_cb_filter.py). Word overlap cũng tính trên
    tiếng Việt để phân tích match với cosine similarity.
    """
    products = pd.read_csv(os.path.join(PROCESSED_DIR, "products_vi.csv"))
    
    product_vectors = scipy.sparse.load_npz(
        os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")
    )
    
    with open(os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")) as f:
        product_id_to_idx = {int(k): v for k, v in json.load(f).items()}
    
    print(f"Products (Vietnamese): {len(products)}")
    print(f"Vectors: {product_vectors.shape}")
    print(f"Mapping: {len(product_id_to_idx)} products")
    
    return products, product_vectors, product_id_to_idx


def tokenize(name):
    """Tokenize tên sản phẩm: lowercase, tách space."""
    if not name or not isinstance(name, str):
        return set()
    return set(name.lower().split())


def compute_word_overlap(name_a, name_b):
    """Tính word overlap giữa 2 tên sản phẩm (tiếng Việt)."""
    tokens_a = tokenize(name_a)
    tokens_b = tokenize(name_b)
    
    if not tokens_a or not tokens_b:
        return 0, 0, 0, 0
    
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = overlap / union if union > 0 else 0
    
    return overlap, len(tokens_a), len(tokens_b), jaccard


def sample_pairs(product_id_to_idx, n_pairs=N_SAMPLE_PAIRS):
    """Random sample các cặp sản phẩm."""
    all_ids = list(product_id_to_idx.keys())
    
    pairs = []
    for _ in range(n_pairs):
        a = random.choice(all_ids)
        b = random.choice(all_ids)
        if a == b:
            continue
        pairs.append((a, b))
    
    return pairs


def compute_similarities(product_vectors, product_id_to_idx, pairs, products):
    """Tính similarity + word overlap cho từng cặp."""
    similarities = []
    valid_pairs = []
    word_overlaps = []
    word_counts_a = []
    word_counts_b = []
    jaccards = []
    
    # Dùng product_name_vi (tiếng Việt) để match với TF-IDF vectors
    pid_to_name = dict(zip(products['product_id'], products['product_name_vi']))
    
    n_total = len(pairs)
    for i, (a, b) in enumerate(pairs):
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{n_total}")
        
        if a not in product_id_to_idx or b not in product_id_to_idx:
            continue
        
        idx_a = product_id_to_idx[a]
        idx_b = product_id_to_idx[b]
        
        sim = cb_similarity(product_vectors, idx_a, [idx_b])[0]
        similarities.append(sim)
        valid_pairs.append((a, b))
        
        # Word overlap trên tên tiếng Việt
        name_a = pid_to_name.get(a, '')
        name_b = pid_to_name.get(b, '')
        overlap, na, nb, jac = compute_word_overlap(name_a, name_b)
        word_overlaps.append(overlap)
        word_counts_a.append(na)
        word_counts_b.append(nb)
        jaccards.append(jac)
    
    return (
        np.array(similarities),
        valid_pairs,
        np.array(word_overlaps),
        np.array(word_counts_a),
        np.array(word_counts_b),
        np.array(jaccards),
    )


def plot_histogram(similarities, save_path):
    """Vẽ histogram phân bố similarity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram tổng thể
    ax1 = axes[0]
    ax1.hist(similarities, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Số cặp')
    ax1.set_title(f'Phân bố CB Similarity (n={len(similarities):,})')
    ax1.axvline(np.median(similarities), color='red', linestyle='--', label=f'Median={np.median(similarities):.3f}')
    ax1.axvline(np.mean(similarities), color='orange', linestyle='--', label=f'Mean={np.mean(similarities):.3f}')
    ax1.legend()
    
    for p, c in [(90, 'green'), (95, 'purple'), (99, 'brown')]:
        val = np.percentile(similarities, p)
        ax1.axvline(val, color=c, linestyle=':', alpha=0.7, label=f'P{p}={val:.3f}')
    ax1.legend(fontsize=8)
    
    # Zoom vào vùng >= 0.3 — nơi substitute xuất hiện
    ax2 = axes[1]
    mask_high = similarities >= 0.3
    ax2.hist(similarities[mask_high], bins=50, color='coral', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Số cặp')
    ax2.set_title(f'Zoom: similarity >= 0.3 (n={mask_high.sum():,})')
    ax2.axvline(0.7, color='red', linestyle='--', alpha=0.7, label='0.7')
    ax2.axvline(0.8, color='darkred', linestyle='--', alpha=0.7, label='0.8')
    ax2.axvline(0.9, color='maroon', linestyle='--', alpha=0.7, label='0.9')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Histogram saved: {save_path}")


def print_overlap_analysis(similarities, word_overlaps, word_counts_a, word_counts_b, jaccards):
    """
    In phân tích word overlap theo từng threshold bucket.
    Giúp trả lời: threshold = 0.8 thì tương ứng trùng bao nhiêu từ?
    """
    print("\n" + "=" * 100)
    print("  PHÂN TÍCH WORD OVERLAP THEO THRESHOLD (similarity >= threshold)")
    print("=" * 100)
    
    header = f"{'threshold':>9} | {'% pairs':>7} | {'avg overlap':>10} | {'avg jaccard':>10} | {'avg len A':>9} | {'avg len B':>9} | {'n pairs':>7}"
    print(header)
    print("-" * 100)
    
    thresholds = np.arange(0.0, 1.025, 0.025)
    for t in thresholds:
        mask = similarities >= t
        n = mask.sum()
        pct = n / len(similarities) * 100
        
        if n > 0:
            avg_overlap = word_overlaps[mask].mean()
            avg_jac = jaccards[mask].mean()
            avg_na = word_counts_a[mask].mean()
            avg_nb = word_counts_b[mask].mean()
        else:
            avg_overlap = 0
            avg_jac = 0
            avg_na = 0
            avg_nb = 0
        
        print(f"{t:>9.3f} | {pct:>6.2f}% | {avg_overlap:>10.2f} | {avg_jac:>10.4f} | {avg_na:>9.2f} | {avg_nb:>9.2f} | {n:>7}")


def print_examples_by_threshold_bucket(similarities, pairs, word_overlaps, jaccards, products, buckets=None):
    """
    In ví dụ cụ thể cho từng bucket threshold để user thấy
    "threshold ~0.8 trùng mấy từ, có hợp lý không".
    """
    if buckets is None:
        buckets = [(0.85, 1.01), (0.75, 0.85), (0.65, 0.75), (0.5, 0.65), (0.3, 0.5), (0.0, 0.3)]
    
    pid_to_name = dict(zip(products['product_id'], products['product_name_vi']))
    n_show = 5
    
    print("\n" + "=" * 100)
    print("  VÍ DỤ CỤ THỂ THEO TỪNG KHOẢNG THRESHOLD")
    print("=" * 100)
    
    for lo, hi in buckets:
        mask = (similarities >= lo) & (similarities < hi)
        indices = np.where(mask)[0]
        n = len(indices)
        
        print(f"\n--- Threshold [{lo:.2f}, {hi:.2f}) — {n} cặp ---")
        
        if n == 0:
            print("  (không có cặp nào)")
            continue
        
        # Lấy mẫu, ưu tiên similarity cao nhất trong bucket
        if n <= n_show:
            sample = indices
        else:
            # Chọn đều: vài cái đầu + vài cái cuối bucket
            sorted_idx = np.argsort(similarities[indices])[::-1]
            sample = indices[sorted_idx[:n_show]]
        
        for idx in sample:
            a, b = pairs[idx]
            sim = similarities[idx]
            overlap = word_overlaps[idx]
            jac = jaccards[idx]
            name_a = pid_to_name.get(a, '?')
            name_b = pid_to_name.get(b, '?')
            tokens_a = tokenize(name_a)
            tokens_b = tokenize(name_b)
            union = tokens_a | tokens_b
            
            print(f"  sim={sim:.4f} | overlap={overlap:.0f}/{len(union)} từ | jaccard={jac:.3f}")
            print(f"    A: {name_a}")
            print(f"    B: {name_b}")
            if overlap > 0:
                common = tokens_a & tokens_b
                print(f"    trùng: {common}")


def print_examples(similarities, pairs, products, n=TOP_K_SHOW):
    """In ví dụ các cặp giống nhất."""
    pid_to_name = dict(zip(products['product_id'], products['product_name_vi']))
    
    # Top-N giống nhất → substitute rõ ràng
    top_indices = np.argsort(similarities)[::-1][:n]
    print(f"\n--- Top-{n} cặp GIỐNG NHAU NHẤT (substitute risk cao) ---")
    print(f"{'#':<4} {'Sim':<8} {'Product A':<45} {'Product B':<45}")
    print("-" * 102)
    for rank, idx in enumerate(top_indices, 1):
        a, b = pairs[idx]
        sim = similarities[idx]
        name_a = pid_to_name.get(a, '?')[:43]
        name_b = pid_to_name.get(b, '?')[:43]
        print(f"{rank:<4} {sim:<8.4f} {name_a:<45} {name_b:<45}")


def main():
    print("=" * 60)
    print("  ĐÁNH GIÁ PHÂN BỐ CB SIMILARITY + WORD OVERLAP")
    print("=" * 60)
    
    # Load
    print("\n1. Loading data...")
    products, product_vectors, product_id_to_idx = load_data()
    
    # Sample pairs
    print(f"\n2. Sampling {N_SAMPLE_PAIRS:,} pairs...")
    pairs = sample_pairs(product_id_to_idx, N_SAMPLE_PAIRS)
    print(f"   Sampled: {len(pairs)} pairs")
    
    # Compute
    print(f"\n3. Computing similarities + word overlap...")
    (similarities, valid_pairs,
     word_overlaps, word_counts_a, word_counts_b,
     jaccards) = compute_similarities(
        product_vectors, product_id_to_idx, pairs, products
    )
    print(f"   Computed: {len(similarities)} pairs")
    
    # Output dir
    save_dir = os.path.join(RESULT_DIR, "cb_similarity_distribution")
    os.makedirs(save_dir, exist_ok=True)
    
    # Stats
    print_overlap_analysis(similarities, word_overlaps, word_counts_a, word_counts_b, jaccards)
    
    # Histogram
    print(f"\n4. Plotting...")
    plot_histogram(similarities, os.path.join(save_dir, "histogram.png"))
    
    # Examples
    print_examples(similarities, valid_pairs, products)
    print_examples_by_threshold_bucket(similarities, valid_pairs, word_overlaps, jaccards, products)
    
    # Save raw data
    df = pd.DataFrame({
        'product_a_id': [p[0] for p in valid_pairs],
        'product_b_id': [p[1] for p in valid_pairs],
        'similarity': similarities,
        'word_overlap': word_overlaps,
        'word_count_a': word_counts_a,
        'word_count_b': word_counts_b,
        'jaccard': jaccards,
    })
    df.to_csv(os.path.join(save_dir, "cb_similarity_samples.csv"), index=False)
    print(f"\n5. Raw data saved: {save_dir}/cb_similarity_samples.csv")
    
    print(f"\n{'=' * 60}")
    print(f"  HOÀN TẤT!")
    print(f"  Dùng bảng word overlap để chọn ENS_CB_THRESHOLD")
    print(f"  Nhìn vào cột 'avg overlap': threshold bao nhiêu thì trùng bấy nhiêu từ?")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()