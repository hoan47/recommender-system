"""
CB Evaluator — Khảo sát và đánh giá Content-Based Diversity Filter.

Các chức năng:
1. survey_similarity_distribution(): Phân bố cosine similarity tổng thể (ngẫu nhiên)
2. survey_candidate_similarity(): Phân bố cosine similarity trên candidate thực tế (từ ensemble)
3. threshold_sweep(): Khảo sát ảnh hưởng của threshold lên % candidate bị loại
4. manual_inspection_samples(): Lấy mẫu theo bins để kiểm tra thủ công
5. export_llm_survey(): Export mẫu cho LLM đánh giá (vùng biên threshold)
"""
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')        # non-interactive backend, chạy không cần display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULT_DIR, CB_THRESHOLD
from src.features.vectorizer import cb_similarity


# ============================================================
# 1. PHÂN BỐ COSINE SIMILARITY TỔNG THỂ
# ============================================================

def survey_similarity_distribution(
    cbfilter,
    products_df,
    n_samples: int = 10000,
    seed: int = 42,
    save_dir: str = None
) -> dict:
    """
    Khảo sát phân bố cosine similarity tổng thể.
    Chọn n_samples cặp sản phẩm ngẫu nhiên, tính similarity.

    Args:
        cbfilter: CBFilter instance (đã fit / có product_vectors + product_id_to_idx)
        products_df: DataFrame [product_id, product_name, aisle_id, department_id]
        n_samples: số lượng mẫu (cặp) ngẫu nhiên
        seed: random seed
        save_dir: đường dẫn lưu ảnh (None = không lưu)

    Returns:
        dict: thống kê (mean, median, std, percentiles, ...)
    """
    rng = random.Random(seed)
    product_ids = list(cbfilter.product_id_to_idx.keys())
    n_products = len(product_ids)

    if n_products < 2:
        print("WARNING: Không đủ sản phẩm để khảo sát.")
        return {}

    similarities = []
    sampled_pairs = []

    print(f"\n{'='*60}")
    print(f"  KHẢO SÁT 1: PHÂN BỐ SIMILARITY TỔNG THỂ")
    print(f"  Số mẫu: {n_samples}")
    print(f"{'='*60}")

    for _ in range(n_samples):
        a_id = rng.choice(product_ids)
        b_id = rng.choice(product_ids)
        while b_id == a_id:
            b_id = rng.choice(product_ids)

        idx_a = cbfilter.product_id_to_idx[a_id]
        idx_b = cbfilter.product_id_to_idx[b_id]
        sim = cb_similarity(cbfilter.product_vectors, idx_a, [idx_b])[0]
        similarities.append(sim)
        sampled_pairs.append((a_id, b_id, sim))

    similarities = np.array(similarities)

    # Thống kê
    stats = {
        'n_samples': n_samples,
        'mean': float(np.mean(similarities)),
        'median': float(np.median(similarities)),
        'std': float(np.std(similarities)),
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'percentiles': {
            'p5': float(np.percentile(similarities, 5)),
            'p10': float(np.percentile(similarities, 10)),
            'p25': float(np.percentile(similarities, 25)),
            'p50': float(np.percentile(similarities, 50)),
            'p75': float(np.percentile(similarities, 75)),
            'p90': float(np.percentile(similarities, 90)),
            'p95': float(np.percentile(similarities, 95)),
            'p99': float(np.percentile(similarities, 99)),
        },
        'frac_above_threshold': {
            f'≥{t}': float(np.mean(similarities >= t))
            for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        }
    }

    # In thống kê
    print(f"\n  Thống kê similarity scores:")
    print(f"    Mean  = {stats['mean']:.4f}")
    print(f"    Median = {stats['median']:.4f}")
    print(f"    Std   = {stats['std']:.4f}")
    print(f"    Min   = {stats['min']:.4f}")
    print(f"    Max   = {stats['max']:.4f}")
    print(f"  Percentiles:")
    for k, v in stats['percentiles'].items():
        print(f"    {k:>5s} = {v:.4f}")
    print(f"  Tỷ lệ >= threshold:")
    for k, v in stats['frac_above_threshold'].items():
        print(f"    {k:>10s} = {v*100:.2f}%")

    # Vẽ histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(similarities, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2,
               label=f"THRESHOLD={CB_THRESHOLD}")
    ax.axvline(stats['median'], color='orange', linestyle=':', linewidth=1.5,
               label=f"Median={stats['median']:.3f}")
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Số lượng mẫu", fontsize=12)
    ax.set_title(f"Phân bố Cosine Similarity tổng thể (n={n_samples:,})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Chèn text thống kê
    textstr = (
        f"Mean={stats['mean']:.3f}  Median={stats['median']:.3f}\n"
        f"≥0.5={stats['frac_above_threshold']['≥0.5']*100:.1f}%  "
        f"≥0.8={stats['frac_above_threshold']['≥0.8']*100:.1f}%"
    )
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "similarity_distribution.png")
        fig.savefig(path, dpi=150)
        print(f"\n  Đã lưu: {path}")

    plt.close(fig)
    return stats


# ============================================================
# 2. PHÂN BỐ COSINE SIMILARITY TRÊN CANDIDATE THỰC TẾ
# ============================================================

def survey_candidate_similarity(
    cbfilter,
    ensemble_model,
    product_ids: list,
    top_k: int = 100,
    save_dir: str = None
) -> dict:
    """
    Khảo sát phân bố cosine similarity giữa product A và candidate từ ensemble.

    Args:
        cbfilter: CBFilter instance
        ensemble_model: EnsembleModel instance (đã fit)
        product_ids: list product_id đầu vào
        top_k: số candidate lấy từ ensemble (trước CB filter)
        save_dir: đường dẫn lưu ảnh

    Returns:
        dict: thống kê
    """
    all_similarities = []
    per_product_stats = []

    print(f"\n{'='*60}")
    print(f"  KHẢO SÁT 2: CANDIDATE SIMILARITY (TỪ ENSEMBLE)")
    print(f"  Số product đầu vào: {len(product_ids)}, top_k={top_k}")
    print(f"{'='*60}")

    for i, pid in enumerate(product_ids):
        if pid not in cbfilter.product_id_to_idx:
            continue

        idx_a = cbfilter.product_id_to_idx[pid]
        recs = ensemble_model.recommend(pid, use_cb_filter=False)

        if not recs:
            continue

        candidate_ids = [r[0] for r in recs[:top_k]]
        valid_candidates = [
            cid for cid in candidate_ids if cid in cbfilter.product_id_to_idx
        ]

        if not valid_candidates:
            continue

        valid_indices = [cbfilter.product_id_to_idx[cid] for cid in valid_candidates]
        sims = cb_similarity(cbfilter.product_vectors, idx_a, valid_indices)

        all_similarities.extend(sims.tolist())

        per_product_stats.append({
            'product_id': pid,
            'n_candidates': len(valid_candidates),
            'mean_sim': float(np.mean(sims)),
            'max_sim': float(np.max(sims)),
            'p90_sim': float(np.percentile(sims, 90)),
            'n_above_threshold': int(np.sum(sims >= CB_THRESHOLD)),
        })

        if (i + 1) % 20 == 0:
            print(f"    Đã xử lý {i+1}/{len(product_ids)} products...")

    if not all_similarities:
        print("  WARNING: Không có dữ liệu similarity nào!")
        return {}

    all_similarities = np.array(all_similarities)

    # Thống kê
    stats = {
        'n_products': len(per_product_stats),
        'n_pairs': len(all_similarities),
        'mean': float(np.mean(all_similarities)),
        'median': float(np.median(all_similarities)),
        'std': float(np.std(all_similarities)),
        'min': float(np.min(all_similarities)),
        'max': float(np.max(all_similarities)),
        'percentiles': {
            'p5': float(np.percentile(all_similarities, 5)),
            'p25': float(np.percentile(all_similarities, 25)),
            'p50': float(np.percentile(all_similarities, 50)),
            'p75': float(np.percentile(all_similarities, 75)),
            'p95': float(np.percentile(all_similarities, 95)),
        },
        'frac_above_threshold': float(np.mean(all_similarities >= CB_THRESHOLD)),
        'avg_filtered_per_product': float(
            np.mean([p['n_above_threshold'] for p in per_product_stats])
        ),
    }

    print(f"\n  Thống kê similarity trên candidate thực tế:")
    print(f"    Số cặp (A, candidate): {stats['n_pairs']}")
    print(f"    Mean  = {stats['mean']:.4f}")
    print(f"    Median = {stats['median']:.4f}")
    print(f"    Std   = {stats['std']:.4f}")
    print(f"    Max   = {stats['max']:.4f}")
    print(f"    >= threshold={CB_THRESHOLD}: {stats['frac_above_threshold']*100:.2f}%")
    print(f"    TB candidate bị loại/product: {stats['avg_filtered_per_product']:.1f}")

    # Vẽ histogram so sánh 2 phân bố
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram candidate
    axes[0].hist(all_similarities, bins=60, color='coral', edgecolor='white', alpha=0.8)
    axes[0].axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f"THRESHOLD={CB_THRESHOLD}")
    axes[0].set_xlabel("Cosine Similarity", fontsize=11)
    axes[0].set_ylabel("Số lượng cặp (A, candidate)", fontsize=11)
    axes[0].set_title(f"Candidate từ ensemble (n={stats['n_pairs']:,})", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    # Boxplot per-product: % bị loại
    filtered_pcts = [
        p['n_above_threshold'] / max(p['n_candidates'], 1) * 100
        for p in per_product_stats
    ]
    axes[1].boxplot(filtered_pcts, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    axes[1].set_ylabel("% candidate bị loại bởi CB Filter", fontsize=11)
    axes[1].set_title(f"% bị loại / product (TB={np.mean(filtered_pcts):.1f}%)", fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticks([])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "candidate_similarity_distribution.png")
        fig.savefig(path, dpi=150)
        print(f"\n  Đã lưu: {path}")

    plt.close(fig)
    return stats


# ============================================================
# 3. THRESHOLD SWEEP
# ============================================================

def threshold_sweep(
    cbfilter,
    ensemble_model,
    product_ids: list,
    thresholds: list = None,
    top_k: int = 100,
    save_dir: str = None
) -> pd.DataFrame:
    """
    Khảo sát ảnh hưởng của threshold lên % candidate bị loại.

    Args:
        cbfilter: CBFilter instance
        ensemble_model: EnsembleModel instance
        product_ids: list product_id đầu vào
        thresholds: list threshold cần sweep (mặc định: 0.0 đến 0.99)
        top_k: số candidate lấy từ ensemble
        save_dir: đường dẫn lưu ảnh

    Returns:
        DataFrame: [threshold, avg_filtered_pct, ...]
    """
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                      0.9, 0.92, 0.95, 0.97, 0.99]

    print(f"\n{'='*60}")
    print(f"  KHẢO SÁT 3: THRESHOLD SWEEP")
    print(f"  Số product: {len(product_ids)}, top_k={top_k}")
    print(f"  Thresholds: {thresholds}")
    print(f"{'='*60}")

    # Pre-compute: với mỗi product A, lấy list similarity của candidate
    product_sims = {}
    valid_count = 0
    for pid in product_ids:
        if pid not in cbfilter.product_id_to_idx:
            continue
        idx_a = cbfilter.product_id_to_idx[pid]
        recs = ensemble_model.recommend(pid, use_cb_filter=False)
        if not recs:
            continue
        candidate_ids = [r[0] for r in recs[:top_k]]
        valid_candidates = [
            cid for cid in candidate_ids if cid in cbfilter.product_id_to_idx
        ]
        if not valid_candidates:
            continue
        valid_indices = [cbfilter.product_id_to_idx[cid] for cid in valid_candidates]
        sims = cb_similarity(cbfilter.product_vectors, idx_a, valid_indices)
        product_sims[pid] = sims
        valid_count += 1

    print(f"  Số product hợp lệ: {valid_count}")

    if not product_sims:
        print("  WARNING: Không có dữ liệu!")
        return pd.DataFrame()

    # Với mỗi threshold, tính % bị loại trung bình
    results = []
    for th in thresholds:
        filtered_counts = []
        for pid, sims in product_sims.items():
            n_filtered = int(np.sum(sims >= th))
            filtered_counts.append(n_filtered / max(len(sims), 1) * 100)

        results.append({
            'threshold': th,
            'avg_filtered_pct': float(np.mean(filtered_counts)),
            'median_filtered_pct': float(np.median(filtered_counts)),
            'std_filtered_pct': float(np.std(filtered_counts)),
            'max_filtered_pct': float(np.max(filtered_counts)),
            'min_filtered_pct': float(np.min(filtered_counts)),
        })

        print(f"    threshold={th:.2f}: avg_filtered={results[-1]['avg_filtered_pct']:.1f}%"
              f"  median={results[-1]['median_filtered_pct']:.1f}%")

    df = pd.DataFrame(results)

    # Vẽ đồ thị
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['threshold'], df['avg_filtered_pct'], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Trung bình')
    ax.fill_between(df['threshold'],
                     df['avg_filtered_pct'] - df['std_filtered_pct'],
                     df['avg_filtered_pct'] + df['std_filtered_pct'],
                     alpha=0.2, color='steelblue', label='±1 Std')
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax.axvline(CB_THRESHOLD, color='red', linestyle='--', linewidth=2,
               label=f"Current THRESHOLD={CB_THRESHOLD}")

    # Đánh dấu threshold tại 10%, 20%, 30% filtered
    for pct in [10, 20, 30]:
        # Tìm threshold gần nhất
        idx = (df['avg_filtered_pct'] - pct).abs().idxmin()
        th_at_pct = df.loc[idx, 'threshold']
        ax.annotate(f"{pct}% @ th={th_at_pct:.2f}",
                    xy=(th_at_pct, pct),
                    xytext=(th_at_pct + 0.08, pct + 5),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9, color='gray')

    ax.set_xlabel("Threshold (cosine similarity)", fontsize=12)
    ax.set_ylabel("% candidate bị loại (trung bình)", fontsize=12)
    ax.set_title(f"Threshold Sweep: % Candidate bị loại vs Threshold\n"
                 f"({valid_count} products, top-{top_k} candidate/product)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "threshold_sweep.png")
        fig.savefig(path, dpi=150)
        print(f"\n  Đã lưu: {path}")
        # Lưu CSV kết quả
        csv_path = os.path.join(save_dir, "threshold_sweep.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Đã lưu: {csv_path}")

    plt.close(fig)
    return df


# ============================================================
# 4. MANUAL INSPECTION SAMPLES
# ============================================================

def manual_inspection_samples(
    cbfilter,
    products_df: pd.DataFrame,
    n_per_bin: int = 10,
    seed: int = 42,
    save_dir: str = None,
    bins: list = None
) -> pd.DataFrame:
    """
    Lấy mẫu cặp sản phẩm theo bins similarity để kiểm tra thủ công.
    Mỗi bin lấy n_per_bin cặp.

    Args:
        cbfilter: CBFilter instance
        products_df: DataFrame [product_id, product_name, aisle_id, department_id, ...]
        n_per_bin: số cặp lấy cho mỗi bin
        seed: random seed
        save_dir: đường dẫn lưu file
        bins: list bin edges (mặc định: [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    Returns:
        DataFrame các mẫu để kiểm tra thủ công
    """
    if bins is None:
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    rng = random.Random(seed)
    product_ids = list(cbfilter.product_id_to_idx.keys())

    # Tạo map product_id → name, aisle, department
    pid2name = dict(zip(products_df['product_id'], products_df['product_name']))
    pid2aisle = dict(zip(products_df['product_id'], products_df['aisle_id']))
    pid2dept = dict(zip(products_df['product_id'], products_df['department_id']))

    # Map id → name cho aisle và department
    aisle_names = {}
    if 'aisle' in products_df.columns:
        aisle_names = dict(zip(products_df['aisle_id'], products_df['aisle']))
    dept_names = {}
    if 'department' in products_df.columns:
        dept_names = dict(zip(products_df['department_id'], products_df['department']))

    # Tạo nhiều mẫu ngẫu nhiên, phân vào bins
    all_samples = []
    max_tries = 100000
    tries = 0

    while len([s for s in all_samples if s is not None]) < n_per_bin * len(bins) - 1:
        if tries > max_tries:
            break
        tries += 1

        a_id = rng.choice(product_ids)
        b_id = rng.choice(product_ids)
        while b_id == a_id:
            b_id = rng.choice(product_ids)

        idx_a = cbfilter.product_id_to_idx[a_id]
        idx_b = cbfilter.product_id_to_idx[b_id]
        sim = cb_similarity(cbfilter.product_vectors, idx_a, [idx_b])[0]

        # Tìm bin
        bin_idx = None
        for i in range(len(bins) - 1):
            if bins[i] <= sim < bins[i + 1]:
                bin_idx = i
                break
        if sim == 1.0:
            bin_idx = len(bins) - 2  # bin cuối

        if bin_idx is None:
            continue

        # Đếm số mẫu đã có trong bin này
        n_in_bin = sum(1 for s in all_samples if s is not None and s['bin_idx'] == bin_idx)
        if n_in_bin >= n_per_bin:
            continue

        all_samples.append({
            'bin_idx': bin_idx,
            'bin_label': f"[{bins[bin_idx]}, {bins[bin_idx+1]})",
            'product_A_id': a_id,
            'product_B_id': b_id,
            'cosine_similarity': float(sim),
            'product_A_name': pid2name.get(a_id, '?'),
            'product_B_name': pid2name.get(b_id, '?'),
            'aisle_A': aisle_names.get(pid2aisle.get(a_id, -1), str(pid2aisle.get(a_id, '?'))),
            'aisle_B': aisle_names.get(pid2aisle.get(b_id, -1), str(pid2aisle.get(b_id, '?'))),
            'dept_A': dept_names.get(pid2dept.get(a_id, -1), str(pid2dept.get(a_id, '?'))),
            'dept_B': dept_names.get(pid2dept.get(b_id, -1), str(pid2dept.get(b_id, '?'))),
        })

    # Loại bỏ None
    samples = [s for s in all_samples if s is not None]
    df = pd.DataFrame(samples)

    print(f"\n{'='*60}")
    print(f"  KHẢO SÁT 4: MANUAL INSPECTION SAMPLES")
    print(f"  Số mẫu mỗi bin: {n_per_bin}")
    print(f"{'='*60}")

    for bin_idx in range(len(bins) - 1):
        bin_label = f"[{bins[bin_idx]}, {bins[bin_idx+1]})"
        bin_samples = df[df['bin_idx'] == bin_idx]
        print(f"\n  Bin {bin_label} ({len(bin_samples)} samples):")
        print(f"  {'A_ID':>6s} | {'A_Name':<35s} | {'B_ID':>6s} | {'B_Name':<35s} | {'Sim':>5s}")
        print(f"  {'-'*6}-+-{'-'*35}-+-{'-'*6}-+-{'-'*35}-+-{'-'*5}")
        for _, row in bin_samples.iterrows():
            a_name = row['product_A_name'][:34] if len(str(row['product_A_name'])) > 34 else row['product_A_name']
            b_name = row['product_B_name'][:34] if len(str(row['product_B_name'])) > 34 else row['product_B_name']
            print(f"  {int(row['product_A_id']):>6d} | {a_name:<35s} | "
                  f"{int(row['product_B_id']):>6d} | {b_name:<35s} | {row['cosine_similarity']:.3f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "manual_samples.csv")
        df.to_csv(path, index=False)
        print(f"\n  Đã lưu: {path}")

    return df


# ============================================================
# 5. EXPORT LLM SURVEY (vùng biên threshold)
# ============================================================

def export_llm_survey(
    cbfilter,
    products_df: pd.DataFrame,
    n_samples: int = 200,
    sim_min: float = 0.2,
    sim_max: float = 0.5,
    seed: int = 42,
    save_dir: str = None
) -> pd.DataFrame:
    """
    Export mẫu ở vùng biên similarity (0.2-0.5) cho LLM đánh giá.
    Đây là vùng threshold khó quyết định — LLM sẽ giúp phân loại.

    Args:
        cbfilter: CBFilter instance
        products_df: DataFrame
        n_samples: số mẫu cần export
        sim_min, sim_max: vùng similarity cần lấy mẫu
        seed: random seed
        save_dir: đường dẫn lưu

    Returns:
        DataFrame với các cột: product_A_id, product_A_name, product_B_id,
                               product_B_name, cosine_similarity, ...
    """
    rng = random.Random(seed)
    product_ids = list(cbfilter.product_id_to_idx.keys())

    pid2name = dict(zip(products_df['product_id'], products_df['product_name']))
    pid2aisle = dict(zip(products_df['product_id'], products_df['aisle_id']))
    pid2dept = dict(zip(products_df['product_id'], products_df['department_id']))

    aisle_names = {}
    if 'aisle' in products_df.columns:
        aisle_names = dict(zip(products_df['aisle_id'], products_df['aisle']))
    dept_names = {}
    if 'department' in products_df.columns:
        dept_names = dict(zip(products_df['department_id'], products_df['department']))

    samples = []
    max_tries = n_samples * 20
    tries = 0

    while len(samples) < n_samples and tries < max_tries:
        tries += 1
        a_id = rng.choice(product_ids)
        b_id = rng.choice(product_ids)
        while b_id == a_id:
            b_id = rng.choice(product_ids)

        idx_a = cbfilter.product_id_to_idx[a_id]
        idx_b = cbfilter.product_id_to_idx[b_id]
        sim = cb_similarity(cbfilter.product_vectors, idx_a, [idx_b])[0]

        if sim_min <= sim < sim_max:
            samples.append({
                'product_A_id': a_id,
                'product_B_id': b_id,
                'cosine_similarity': float(sim),
                'product_A_name': pid2name.get(a_id, '?'),
                'product_B_name': pid2name.get(b_id, '?'),
                'aisle_A': aisle_names.get(pid2aisle.get(a_id, -1), ''),
                'aisle_B': aisle_names.get(pid2aisle.get(b_id, -1), ''),
                'dept_A': dept_names.get(pid2dept.get(a_id, -1), ''),
                'dept_B': dept_names.get(pid2dept.get(b_id, -1), ''),
            })

    df = pd.DataFrame(samples)

    print(f"\n{'='*60}")
    print(f"  KHẢO SÁT 5: LLM SURVEY SAMPLES")
    print(f"  Vùng similarity: [{sim_min}, {sim_max})")
    print(f"  Số mẫu: {len(df)}")
    print(f"{'='*60}")

    print(f"\n  Mẫu (10 cặp đầu tiên):")
    print(f"  {'A_ID':>6s} | {'A_Name':<30s} | {'B_ID':>6s} | {'B_Name':<30s} | {'Sim':>5s}")
    print(f"  {'-'*6}-+-{'-'*30}-+-{'-'*6}-+-{'-'*30}-+-{'-'*5}")
    for _, row in df.head(10).iterrows():
        a_name = str(row['product_A_name'])[:29]
        b_name = str(row['product_B_name'])[:29]
        print(f"  {int(row['product_A_id']):>6d} | {a_name:<30s} | "
              f"{int(row['product_B_id']):>6d} | {b_name:<30s} | {row['cosine_similarity']:.3f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "llm_survey_samples.csv")
        df.to_csv(path, index=False)
        print(f"\n  Đã lưu: {path}")

    return df