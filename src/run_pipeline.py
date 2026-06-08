"""
Pipeline hoàn chỉnh để build và evaluate model 2 (recommender-system) với các cải tiến

Các cải tiến so với phiên bản gốc:
  1. ✅ Cross-dept bonus trong Confidence scoring
  2. ✅ Reorder rate bonus trong Confidence scoring
  3. ✅ Giảm CB threshold từ 0.85 → 0.45
  4. ✅ Diversity filter (đa dạng department)
  5. ✅ Department direction filter

Chạy:
    python src/run_pipeline.py

Output:
    - models/confidence_matrix.npz  (đã có cross-dept + reorder bonus)
    - models/hybrid_matrix.npz      (đã có CB filter mới)
    - results/metrics_baseline.csv  (kết quả KHÔNG filter)
    - results/metrics_improved.csv  (kết quả CÓ filter: direction + diversity)
    - results/comparison.csv        (so sánh baseline vs improved)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

from src.config import MODELS_DIR, RESULTS_DIR
from src.data_loader import load_products, load_prior
from src.features.build_association_rules import build_cooc, build_confidence, save as save_conf
from src.features.build_cb import build_cb_vectors
from src.features.build_knowledge_graph import build_dept_spmi, build_graph, train_node2vec, compute_similarity
from src.features.build_hybrid import build_hybrid, save as save_hybrid
from src.features.dept_direction import build_dept_direction, filter_by_direction, dept_suggest
from src.features.diversity import diversity_filter
from src.recommend import set_prod_dept_map, recommend_simple
from src.evaluation.evaluate import run_comparison


def build_reorder_rate(prior_df):
    """Xây dựng product_id → reorder rate mapping."""
    return prior_df.groupby('product_id')['reordered'].mean().to_dict()


def main():
    total_t0 = time.time()
    print("=" * 75)
    print("  PIPELINE: Build + Evaluate")
    print("=" * 75)
    
    # ── Step 1: Load dữ liệu ──────────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 1: Loading data")
    print("-" * 60)
    products_df = load_products()
    prior_df = load_prior()
    print(f"  Products: {len(products_df):,} | Prior: {len(prior_df):,}")
    
    prod_dept_map = dict(zip(products_df['product_id'], products_df['department_id']))
    reorder_rate = build_reorder_rate(prior_df)
    
    # ── Step 2: Build Co-occurrence ───────────────────────
    print("\n" + "-" * 60)
    print("  STEP 2: Building Co-occurrence (Numba JIT)")
    print("-" * 60)
    cooc, freq = build_cooc(prior_df)
    
    # ── Step 3: Build Confidence (với reorder bonus) ──
    print("\n" + "-" * 60)
    print("  STEP 3: Building Confidence (with reorder bonus)")
    print("-" * 60)
    confidence = build_confidence(
        cooc, freq,
        reorder_rate=reorder_rate,
        freq_min=30, top_k=100
    )
    save_conf(cooc, confidence)
    del cooc; gc.collect()
    
    # ── Step 4: Build CB vectors ──────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 4: Building CB vectors")
    print("-" * 60)
    build_cb_vectors(products_df, prior_df)
    
    from src.features.build_cb import cb_vectors
    print(f"  CB vectors: {len(cb_vectors):,}")
    
    # ── Step 5: Build KG ──────────────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 5: Building Knowledge Graph (KG)")
    print("-" * 60)
    n_products = confidence.shape[0]
    dept_spmi, n_depts = build_dept_spmi(prior_df, products_df)
    
    # Dùng SPMI từ confidence matrix (chuyển sang positive)
    from scipy.sparse import csr_matrix
    spmi = confidence.copy()
    spmi.data[spmi.data < 0] = 0
    spmi = spmi.maximum(spmi.T)  # Đối xứng hóa
    
    G = build_graph(spmi, dept_spmi, products_df, n_depts)
    del spmi, dept_spmi; gc.collect()
    
    emb = train_node2vec(G, n_products, dim=64, walk_len=20, n_walks=50, epochs=1)
    kg_sim = compute_similarity(emb, top_k=100)
    del G, emb; gc.collect()
    
    # ── Step 6: Build Hybrid (với CB threshold mới 0.45) ──
    print("\n" + "-" * 60)
    print("  STEP 6: Building Hybrid matrix (CB threshold=0.45)")
    print("-" * 60)
    hybrid = build_hybrid(
        confidence, kg_sim, cb_vectors,
        alpha=0.2, beta=0.8, cb_thresh=0.45
    )
    save_hybrid(hybrid)
    del kg_sim; gc.collect()
    
    # ── Step 7: Build Department Direction ────────────────
    print("\n" + "-" * 60)
    print("  STEP 7: Building Department Direction")
    print("-" * 60)
    build_dept_direction(prior_df, prod_dept_map)
    
    # ── Step 8: Set prod_dept_map cho recommend module ────
    set_prod_dept_map(prod_dept_map)
    
    # ── Step 9: Evaluate ──────────────────────────────────
    print("\n" + "-" * 60)
    print("  STEP 9: Evaluating models")
    print("-" * 60)
    
    # Tạo test cases từ data_loader
    from src.data_loader import load_temporal_test_cases
    test_cases, model_df = load_temporal_test_cases(train_ratio=0.8)
    
    # Import evaluate module
    from src.evaluation.evaluate import evaluate_one, calc_metrics
    
    from src.recommend import recommend as recommend_filtered
    
    from tqdm import tqdm
    from collections import defaultdict
    
    ks = (5, 10, 20)
    
    # Baseline: Confidence (không filter)
    print("\n  Evaluating BASELINE (Confidence, no filter)...")
    baseline_metrics = defaultdict(list)
    for seed, truth in tqdm(test_cases, desc="  [Baseline]", ncols=80):
        if not truth:
            continue
        recs = recommend_simple(seed, max(ks))
        if not recs:
            continue
        m = calc_metrics(recs, truth, ks)
        for key, val in m.items():
            baseline_metrics[key].append(val)
    
    baseline_result = {}
    for key, vals in baseline_metrics.items():
        baseline_result[key] = round(float(np.mean(vals)), 4)
    baseline_result['N_valid'] = len(next(iter(baseline_metrics.values())))
    
    print(f"\n  BASELINE RESULTS:")
    for k in ks:
        print(f"    H@{k} = {baseline_result.get(f'H@{k}', 0):.4f} | "
              f"NDCG@{k} = {baseline_result.get(f'NDCG@{k}', 0):.4f} | "
              f"MAP@{k} = {baseline_result.get(f'MAP@{k}', 0):.4f}")
    
    # Improved
    from src.recommend import _load_matrices
    _load_matrices()  # Đảm bảo matrices đã load
    
    print("\n  Evaluating IMPROVED (direction + diversity filter)...")
    improved_metrics = defaultdict(list)
    for seed, truth in tqdm(test_cases, desc="  [Improved]", ncols=80):
        if not truth:
            continue
        recs = recommend_filtered(seed, max(ks))
        if not recs:
            continue
        m = calc_metrics(recs, truth, ks)
        for key, val in m.items():
            improved_metrics[key].append(val)
    
    improved_result = {}
    for key, vals in improved_metrics.items():
        improved_result[key] = round(float(np.mean(vals)), 4)
    improved_result['N_valid'] = len(next(iter(improved_metrics.values())))
    
    print(f"\n  IMPROVED RESULTS (direction + diversity):")
    for k in ks:
        print(f"    H@{k} = {improved_result.get(f'H@{k}', 0):.4f} | "
              f"NDCG@{k} = {improved_result.get(f'NDCG@{k}', 0):.4f} | "
              f"MAP@{k} = {improved_result.get(f'MAP@{k}', 0):.4f}")
    
    # ── Comparison ────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  COMPARISON: Baseline vs Improved")
    print("=" * 95)
    
    print(f"\n  {'Metric':<15} {'Baseline':<12} {'Improved':<12} {'Delta':<12} {'Change':<12}")
    print("  " + "-" * 63)
    for k in ks:
        for prefix in ['H', 'NDCG', 'MAP']:
            col = f'{prefix}@{k}'
            base_val = baseline_result.get(col, 0)
            impr_val = improved_result.get(col, 0)
            delta = impr_val - base_val
            pct = (delta / base_val * 100) if base_val > 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"  {col:<15} {base_val:<12.4f} {impr_val:<12.4f} {sign}{delta:<11.4f} {sign}{pct:.1f}%")
    
    # ── Save results ──────────────────────────────────────
    results_df = pd.DataFrame({
        'Metric': [f'H@{k}' for k in ks] + [f'NDCG@{k}' for k in ks] + [f'MAP@{k}' for k in ks],
        'Baseline': [baseline_result.get(f'H@{k}', 0) for k in ks] + 
                     [baseline_result.get(f'NDCG@{k}', 0) for k in ks] +
                     [baseline_result.get(f'MAP@{k}', 0) for k in ks],
        'Improved': [improved_result.get(f'H@{k}', 0) for k in ks] + 
                     [improved_result.get(f'NDCG@{k}', 0) for k in ks] +
                     [improved_result.get(f'MAP@{k}', 0) for k in ks],
    })
    results_df['Delta'] = results_df['Improved'] - results_df['Baseline']
    
    comp_file = RESULTS_DIR / "comparison.csv"
    results_df.to_csv(comp_file, index=False, encoding='utf-8-sig')
    print(f"\n  Saved comparison: {comp_file}")
    
    # Lưu metrics riêng
    bl_file = RESULTS_DIR / "metrics_baseline.csv"
    pd.DataFrame([baseline_result]).to_csv(bl_file, index=False)
    
    impr_file = RESULTS_DIR / "metrics_improved.csv"
    pd.DataFrame([improved_result]).to_csv(impr_file, index=False)
    
    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 75}")
    print(f"  PIPELINE COMPLETE! Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()