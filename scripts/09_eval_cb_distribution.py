"""
09 — Đánh giá CB similarity theo số từ trùng nhau.
So sánh 3 loại similarity: TF-IDF, Count Vectorizer (Overlap), Ensemble (alpha * Overlap + (1-alpha) * TF-IDF).
Với mỗi overlap = 1..10, tìm N_EXAMPLES cặp có đúng N từ chung, tính similarity.

Vẽ đồ thị 3 đường (mean ± std) + scatter subset để hiển thị distribution.
"""
import json
import os
import sys
import random
import math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import MODEL_DIR, PROCESSED_DIR, RESULT_DIR, CB_ALPHA
from src.features.vectorizer import cb_similarity

# ─────────────────────────────────────────────────────────────
# TUỲ CHỈNH — thoải mái sửa sau
# ─────────────────────────────────────────────────────────────
N_EXAMPLES =   1000      # số cặp cho mỗi overlap (đặt 1K, có thể giảm nếu chậm)
MAX_OVERLAP = 10         # từ 1 đến 10 từ trùng
MAX_BUCKET_TRIES = 5_000_000  # giới hạn lấy mẫu trong bucket (tránh treo vô hạn)

# Scatter subset: chỉ vẽ 1/SUBSAMPLE_FRAC số điểm để đỡ rát
SUBSAMPLE_FRAC = 0.02    # vẽ 2% số điểm → ~4000 điểm/overlap với 200K
SCATTER_ALPHA = 0.05

SAVE_IMAGE = os.path.join(RESULT_DIR, "cb_similarity_distribution",
                          "cb_overlap_distribution.png")
SAVE_HIST_IMAGE = os.path.join(RESULT_DIR, "cb_similarity_distribution",
                               "cb_score_bins_histogram.png")
# Số bins score từ 0.0 đến 1.0
SCORE_BINS = np.arange(0.0, 1.05, 0.1)  # [0.0, 0.1, ..., 1.0]
# ─────────────────────────────────────────────────────────────


def load_data():
    products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))

    tfidf_vectors = scipy.sparse.load_npz(
        os.path.join(MODEL_DIR, "cb_filter", "tfidf_vectors.npz")
    )
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
    """Overlap Score: |A ∩ B| / min(|A|, |B|) trên nhị phân hoá."""
    vec_a = count_vectors[idx_a]
    vecs_b = count_vectors[idx_b_list]

    bin_a = vec_a.astype(bool).astype(np.float64)
    bin_b = vecs_b.astype(bool).astype(np.float64)

    intersection = bin_b.dot(bin_a.T).toarray().ravel()
    sum_a = bin_a.sum()
    sum_b = np.array(bin_b.sum(axis=1)).ravel()
    denom = np.minimum(sum_a, sum_b)

    return np.divide(intersection, denom,
                     out=np.zeros_like(intersection, dtype=float),
                     where=denom != 0)


def build_inverted_index(products, product_id_to_idx):
    """
    Xây inverted index: token -> list of (product_id, idx)
    Chỉ giữ các product có trong product_id_to_idx.
    """
    inv = defaultdict(list)
    pid_to_idx = product_id_to_idx
    for _, row in products.iterrows():
        pid = row['product_id']
        if pid not in pid_to_idx:
            continue
        name = row.get('product_name', '')
        tokens = tokenize(name)
        for t in tokens:
            inv[t].append(pid)
    return inv


def find_pairs_by_overlap_fast(products, tfidf_vectors, count_vectors,
                                 product_id_to_idx):
    """
    Dùng inverted index để tìm nhanh các cặp có overlap >= 1.
    Nhóm vào bucket overlap=1..MAX_OVERLAP, mỗi bucket N_EXAMPLES cặp.
    """
    alpha = CB_ALPHA
    pid_to_name = dict(zip(products['product_id'], products['product_name']))
    pid_to_idx = product_id_to_idx
    all_pids = list(pid_to_idx.keys())

    print("  Xây inverted index...")
    inv = build_inverted_index(products, product_id_to_idx)
    print(f"    {len(inv)} tokens, "
          f"{(sum(len(v) for v in inv.values())):,} entries")

    # Với mỗi product, pre-compute tokens để tính overlap nhanh
    print("  Pre-compute tokens cho từng product...")
    pid_tokens = {}
    for pid in all_pids:
        name = pid_to_name.get(pid, '')
        pid_tokens[pid] = tokenize(name)

    # buckets: overlap -> set of frozenset({a, b}) để tránh trùng cặp
    buckets = {n: set() for n in range(1, MAX_OVERLAP + 1)}
    needed = set(range(1, MAX_OVERLAP + 1))

    # Duyệt: random product A, lấy candidates từ inverted index của tokens A
    # Trộn all_pids để không bị bias
    print("  Bắt đầu sampling...")
    random.shuffle(all_pids)
    total_found = 0
    target_total = N_EXAMPLES * MAX_OVERLAP

    for idx_a, pid_a in enumerate(all_pids):
        if not needed:
            break
        if idx_a % 500 == 0:
            filled = sum(len(v) for v in buckets.values())
            print(f"    product {idx_a}/{len(all_pids)}, "
                  f"đã có {filled:,}/{target_total:,} cặp, "
                  f"còn thiếu overlap {sorted(needed)}")

        tokens_a = pid_tokens[pid_a]
        if not tokens_a:
            continue

        # Gom candidates từ các token của A
        cand_set = set()
        for t in tokens_a:
            cand_set.update(inv[t])
        cand_set.discard(pid_a)

        # Chỉ giữ candidate có overlap cần thiết
        # Sắp xếp để deterministic hơn (tránh bias thứ tự token)
        candidates = sorted(cand_set)

        for pid_b in candidates:
            if pid_b not in pid_to_idx:
                continue
            if pid_a == pid_b:
                continue
            pair = frozenset([pid_a, pid_b])
            overlap = len(tokens_a & pid_tokens[pid_b])
            if overlap not in needed:
                continue
            if pair in buckets[overlap]:
                continue
            if len(buckets[overlap]) >= N_EXAMPLES:
                needed.discard(overlap)
                continue

            buckets[overlap].add(pair)

            if len(buckets[overlap]) >= N_EXAMPLES:
                needed.discard(overlap)

            filled_now = sum(len(v) for v in buckets.values())
            if filled_now >= target_total:
                break

        if sum(len(v) for v in buckets.values()) >= target_total:
            break

    # Với mỗi cặp trong buckets, tính similarity
    print("  Tính similarity cho các cặp...")
    result = {}
    for overlap in range(1, MAX_OVERLAP + 1):
        pairs = list(buckets.get(overlap, []))
        # Nếu thiếu, random bổ sung
        if len(pairs) < N_EXAMPLES:
            print(f"    overlap={overlap}: chỉ có {len(pairs)} cặp, "
                  f"cần bổ sung...")
            # Bổ sung bằng brute-force
            extra = set()
            all_pids_shuffled = all_pids[:]
            random.shuffle(all_pids_shuffled)
            for pa in all_pids_shuffled:
                if len(pairs) + len(extra) >= N_EXAMPLES:
                    break
                ta = pid_tokens[pa]
                for pb in all_pids_shuffled:
                    if pa == pb:
                        continue
                    pair2 = frozenset([pa, pb])
                    if pair2 in buckets[overlap] or pair2 in extra:
                        continue
                    ov = len(ta & pid_tokens[pb])
                    if ov == overlap:
                        extra.add(pair2)
                        if len(pairs) + len(extra) >= N_EXAMPLES:
                            break
            pairs.extend(list(extra))

        # Lấy ngẫu nhiên đúng N_EXAMPLES
        if len(pairs) > N_EXAMPLES:
            pairs = random.sample(pairs, N_EXAMPLES)

        result[overlap] = []
        for pair in pairs:
            pids = list(pair)
            a, b = pids[0], pids[1]
            name_a = pid_to_name.get(a, '')
            name_b = pid_to_name.get(b, '')
            idx_a = pid_to_idx[a]
            idx_b = pid_to_idx[b]

            sim_tfidf = cb_similarity(tfidf_vectors, idx_a, [idx_b])[0]
            sim_count = _overlap_similarity(count_vectors, idx_a, [idx_b])[0]
            sim_ensemble = alpha * sim_count + (1.0 - alpha) * sim_tfidf
            common = pid_tokens[a] & pid_tokens[b]
            result[overlap].append((sim_tfidf, sim_count, sim_ensemble,
                                    name_a, name_b, sorted(common)))

    return result


def plot_results(data, save_path):
    """Vẽ 3 đường mean ± std + scatter subset."""
    overlaps = list(range(1, MAX_OVERLAP + 1))
    means = {'TF-IDF': [], 'Overlap': [], 'Ensemble': []}
    stds  = {'TF-IDF': [], 'Overlap': [], 'Ensemble': []}
    all_scatter = {'TF-IDF': [], 'Overlap': [], 'Ensemble': [], 'overlap': []}

    for ov in overlaps:
        pairs = data[ov]
        tfidf_vals  = np.array([p[0] for p in pairs])
        count_vals  = np.array([p[1] for p in pairs])
        ensemble_vals = np.array([p[2] for p in pairs])

        means['TF-IDF'].append(np.mean(tfidf_vals))
        means['Overlap'].append(np.mean(count_vals))
        means['Ensemble'].append(np.mean(ensemble_vals))
        stds['TF-IDF'].append(np.std(tfidf_vals))
        stds['Overlap'].append(np.std(count_vals))
        stds['Ensemble'].append(np.std(ensemble_vals))

        # Scatter subset
        n_scatter = max(1, int(len(pairs) * SUBSAMPLE_FRAC))
        idxs = np.random.choice(len(pairs), n_scatter, replace=False)
        all_scatter['TF-IDF'].extend(tfidf_vals[idxs])
        all_scatter['Overlap'].extend(count_vals[idxs])
        all_scatter['Ensemble'].extend(ensemble_vals[idxs])
        all_scatter['overlap'].extend([ov] * n_scatter)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'TF-IDF': '#1f77b4', 'Overlap': '#ff7f0e', 'Ensemble': '#2ca02c'}
    markers = {'TF-IDF': 'o', 'Overlap': 's', 'Ensemble': '^'}

    # Scatter background
    for label, color in colors.items():
        ax.scatter(all_scatter['overlap'], all_scatter[label],
                   s=2, alpha=SCATTER_ALPHA, color=color, label='_nolegend_')

    # Line + error band
    for label in ['TF-IDF', 'Overlap', 'Ensemble']:
        mu = means[label]
        sigma = np.array(stds[label])
        ax.plot(overlaps, mu, color=colors[label], marker=markers[label],
                linewidth=2, label=label, markersize=6)
        ax.fill_between(overlaps,
                         np.array(mu) - sigma,
                         np.array(mu) + sigma,
                         color=colors[label], alpha=0.12)

    ax.set_xlabel('Số từ trùng nhau (Overlap)', fontsize=13)
    ax.set_ylabel('Similarity Score', fontsize=13)
    ax.set_title('CB Similarity theo số từ trùng nhau\n'
                 f'(N={N_EXAMPLES:,} cặp/overlap, Ensemble α={CB_ALPHA})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(overlaps)
    ax.set_xlim(0.5, MAX_OVERLAP + 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Đồ thị đã lưu: {save_path}")


def plot_score_distribution_histogram(data, save_path):
    """
    Vẽ histogram phân phối điểm similarity theo score bins (0.0-0.1, ..., 0.9-1.0).
    Trục Y: Tỷ lệ % cặp sản phẩm đạt mức điểm đó.
    Gom tất cả các overlap lại để có tổng thể.
    """
    tfidf_vals  = []
    count_vals  = []
    ensemble_vals = []

    for ov in range(1, MAX_OVERLAP + 1):
        pairs = data.get(ov, [])
        for p in pairs:
            tfidf_vals.append(p[0])
            count_vals.append(p[1])
            ensemble_vals.append(p[2])

    labels_map = {
        'TF-IDF':    (tfidf_vals,  '#1f77b4'),
        'Overlap':   (count_vals,  '#ff7f0e'),
        'Ensemble':  (ensemble_vals, '#2ca02c'),
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    n_bins = len(SCORE_BINS) - 1
    bar_width = 0.22
    x_centers = np.arange(n_bins)

    for i, (label, (vals, color)) in enumerate(labels_map.items()):
        counts, _ = np.histogram(vals, bins=SCORE_BINS)
        pct = counts / len(vals) * 100
        offset = (i - 1) * bar_width
        ax.bar(x_centers + offset, pct, bar_width,
               color=color, alpha=0.85, label=label,
               edgecolor='white', linewidth=0.5)

    bin_labels = [f"[{SCORE_BINS[j]:.1f}-{SCORE_BINS[j+1]:.1f})"
                  for j in range(n_bins - 1)]
    bin_labels.append(f"[{SCORE_BINS[-2]:.1f}-{SCORE_BINS[-1]:.1f}]")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Score Bins', fontsize=13)
    ax.set_ylabel('Tỷ lệ % cặp sản phẩm', fontsize=13)
    ax.set_title('Phân phối điểm CB Similarity theo Score Bins\n'
                 f'(Tổng {len(tfidf_vals):,} cặp sample, Ensemble α={CB_ALPHA})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(ax.get_ylim()[1], 50))
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Histogram đã lưu: {save_path}")


def print_results(data):
    print()
    print("=" * 140)
    print("  CB SIMILARITY THEO SỐ TỪ TRÙNG (TF-IDF | Overlap | Ensemble)")
    print("=" * 140)

    for overlap in range(1, MAX_OVERLAP + 1):
        pairs = data.get(overlap, [])
        print(f"\n--- overlap = {overlap} ({len(pairs)} cặp, "
              f"hiện 5 cặp đầu) ---")
        print(f"  {'TF-IDF':>8} | {'Overlap':>8} | {'Ensemble':>8} | Sản phẩm A ~ Sản phẩm B | Từ trùng")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+{'-'*45}+-{'-'*30}")
        for sim_t, sim_c, sim_e, name_a, name_b, common in pairs[:5]:
            common_str = ", ".join(common) if common else "(không có)"
            print(f"  {sim_t:>8.4f} | {sim_c:>8.4f} | {sim_e:>8.4f} | "
                  f"\"{name_a}\" ~ \"{name_b}\" | {common_str}")

    print("\n" + "=" * 140)


def main():
    print("=" * 60)
    print("  ĐÁNH GIÁ CB SIMILARITY THEO WORD OVERLAP")
    print(f"  Ensemble: alpha(Overlap)={CB_ALPHA}")
    print(f"  N_EXAMPLES = {N_EXAMPLES:,} cặp/overlap")
    print("=" * 60)

    print("\n1. Loading data...")
    products, tfidf_vectors, count_vectors, product_id_to_idx = load_data()
    print(f"   products={len(products)}")
    print(f"   TF-IDF vectors={tfidf_vectors.shape}")
    print(f"   Count vectors={count_vectors.shape}")

    print(f"\n2. Tìm cặp cho overlap 1->{MAX_OVERLAP}, "
          f"mỗi cái {N_EXAMPLES:,} cặp...")
    data = find_pairs_by_overlap_fast(products, tfidf_vectors, count_vectors,
                                       product_id_to_idx)

    # Tổng kết số lượng
    for ov in range(1, MAX_OVERLAP + 1):
        print(f"   overlap={ov}: {len(data[ov])} cặp")

    print_results(data)

    # Lưu CSV (chỉ sample 10000 dòng để nhẹ)
    save_dir = os.path.join(RESULT_DIR, "cb_similarity_distribution")
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for overlap in range(1, MAX_OVERLAP + 1):
        for sim_t, sim_c, sim_e, name_a, name_b, common in data[overlap]:
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
    # Lưu full
    path_full = os.path.join(save_dir, "cb_overlap_samples_full.csv")
    df.to_csv(path_full, index=False)
    print(f"\nRaw data (full): {path_full}  ({len(df):,} dòng)")

    # Plot
    print("\n3. Vẽ đồ thị overlap...")
    plot_results(data, SAVE_IMAGE)

    print("\n4. Vẽ histogram score bins...")
    plot_score_distribution_histogram(data, SAVE_HIST_IMAGE)

    print("\n  HOÀN TẤT!")


if __name__ == '__main__':
    main()