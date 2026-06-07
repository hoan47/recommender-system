"""
Tune hyperparameters — Grid search tự động 4 phase

Quy trình:
  Phase 0: CB (Content-Based) — tune CB_MIN_DF, CB_MAX_DF, CB_MAX_FEATURES
  Phase 1: SPMI (Collaborative) — tune SPMI_K, SPMI_TOP_K (dùng co-occurrence cache)
  Phase 2: KG (Knowledge Graph) — tune KG_DIM, KG_WALK_LENGTH, KG_NUM_WALKS, KG_EPOCHS
  Phase 3: Hybrid — tune HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH

Mỗi tổ hợp: build model → evaluate → snapshot kết quả.
Chọn best params dựa trên recall@10 (hoặc metric khác).

Output lưu tại: results/tune_results/
  phase0_cb/     — snapshots + best_params.json
  phase1_spmi/   — snapshots + best_params.json
  phase2_kg/     — snapshots + best_params.json
  phase3_hybrid/ — snapshots + best_params.json
  final_best_params.json  — tổng hợp best toàn cục
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import json
import math
import time
import numpy as np
from scipy.sparse import save_npz, load_npz, csr_matrix
from tqdm import tqdm

from src.config import (
    MODELS_DIR, RESULTS_DIR, EVAL_KS,
    CB_MIN_DF, CB_MAX_DF, CB_MAX_FEATURES, TOP_K,
    SPMI_K, TOTAL_PRIOR_ORDERS, SPMI_TOP_K,
    KG_DIM, KG_WALK_LENGTH, KG_NUM_WALKS, KG_EPOCHS,
    HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH,
)
from src.data_loader import load_products, load_prior, load_train_test

# ============================================================
# Imports các module feature (lazy, chỉ import khi cần)
# ============================================================

def _import_build_cb():
    from src.features import build_cb
    return build_cb

def _import_build_spmi():
    from src.features import build_spmi
    return build_spmi

def _import_build_kg():
    from src.features import build_knowledge_graph as kg
    return kg

def _import_build_hybrid():
    from src.features import build_hybrid
    return build_hybrid

from src.evaluation.evaluate import evaluate_model

# ============================================================
# Đường dẫn tune results
# ============================================================
TUNE_DIR = RESULTS_DIR / "tune_results"
PHASE0_DIR = TUNE_DIR / "phase0_cb"
PHASE1_DIR = TUNE_DIR / "phase1_spmi"
PHASE2_DIR = TUNE_DIR / "phase2_kg"
PHASE3_DIR = TUNE_DIR / "phase3_hybrid"

for _d in [TUNE_DIR, PHASE0_DIR, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Metric chính để chọn best params
BEST_METRIC = "recall@10"

# ============================================================
# Utility functions
# ============================================================

def snapshot_name(prefix, params_dict):
    """Tạo tên file snapshot từ params dict."""
    parts = []
    for k, v in params_dict.items():
        # Rút gọn tên tham số: bỏ prefix
        short = k.split("_", 1)[1] if "_" in k else k
        # Làm tròn số float
        if isinstance(v, float):
            v_str = f"{v:.2f}".replace(".", "p")
        else:
            v_str = str(v)
        parts.append(f"{short}{v_str}")
    return f"{prefix}_{'_'.join(parts)}.json"


def save_snapshot(phase_dir, prefix, params, metrics, elapsed):
    """Lưu snapshot kết quả của một tổ hợp params."""
    data = {
        "params": params,
        "metrics": metrics,
        "time_seconds": round(elapsed, 2),
    }
    fname = snapshot_name(prefix, params)
    path = phase_dir / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_best(phase_dir, best_params, best_metrics, best_model_file):
    """Lưu best params của một phase."""
    data = {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_model_file": str(best_model_file),
        "metric_used": BEST_METRIC,
    }
    path = phase_dir / "best_params.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  [Tune] Best params saved: {path}")
    print(f"  [Tune] Best {BEST_METRIC} = {best_metrics.get(BEST_METRIC, 'N/A')}")


def extract_metric(model_metrics, metric=BEST_METRIC):
    """Trích metric từ dict kết quả evaluate."""
    if model_metrics and metric in model_metrics:
        return model_metrics[metric]
    return 0.0


def get_cb_trained_vectors():
    """Lấy cb_vectors từ module build_cb (đã train)."""
    from src.features.build_cb import cb_vectors
    return cb_vectors


# ============================================================
# PHASE 0: Content-Based (CB)
# ============================================================

def tune_cb(products_df, prior_df, train_df):
    """
    Grid search CB params:
      CB_MIN_DF ∈ [3, 5, 10]
      CB_MAX_DF ∈ [0.6, 0.8, 0.95]
      CB_MAX_FEATURES ∈ [5000, 10000, 20000]
    """
    print("\n" + "=" * 60)
    print("  PHASE 0: Tuning Content-Based (CB)")
    print("=" * 60)

    grid_min_df = [3, 5, 10]
    grid_max_df = [0.6, 0.8, 0.95]
    grid_max_feat = [5000, 10000, 20000]

    best_metric_val = -1.0
    best_params = None
    best_metrics = None
    best_cb_vectors = None

    total = len(grid_min_df) * len(grid_max_df) * len(grid_max_feat)
    current = 0

    for min_df in grid_min_df:
        for max_df in grid_max_df:
            for max_feat in grid_max_feat:
                current += 1
                params = {
                    "CB_MIN_DF": min_df,
                    "CB_MAX_DF": max_df,
                    "CB_MAX_FEATURES": max_feat,
                }
                print(f"\n  --- [{current}/{total}] CB: min_df={min_df}, max_df={max_df}, max_feat={max_feat} ---")
                t0 = time.time()

                # Override config values
                import src.config as cfg
                cfg.CB_MIN_DF = min_df
                cfg.CB_MAX_DF = max_df
                cfg.CB_MAX_FEATURES = max_feat

                # Build CB vectors
                build_cb = _import_build_cb()
                build_cb.cb_vectors.clear()
                build_cb.build_cb_vectors(products_df, prior_df)
                cb_vec = get_cb_trained_vectors()
                if not cb_vec:
                    print("  [Tune] CB vectors empty, skip")
                    continue

                # Build CB similarity matrix (dùng hàm từ build_hybrid)
                from src.features.build_hybrid import build_cb_sparse, build_cb_similarity
                n_products = int(prior_df["product_id"].max()) + 1
                cb_feat = build_cb_sparse(cb_vec, n_products)
                if cb_feat.nnz == 0:
                    print("  [Tune] CB feature matrix empty, skip")
                    del cb_feat; gc.collect()
                    continue
                cb_sim = build_cb_similarity(cb_feat, top_k=TOP_K)
                del cb_feat; gc.collect()

                # Evaluate
                metrics = evaluate_model(cb_sim, train_df, name=f"CB(min_df={min_df})", ks=EVAL_KS)
                elapsed = time.time() - t0

                # Snapshot
                combined_metrics = {"CB": metrics}
                save_snapshot(PHASE0_DIR, "cb", params, combined_metrics, elapsed)

                metric_val = extract_metric(metrics)
                if metric_val > best_metric_val:
                    best_metric_val = metric_val
                    best_params = params.copy()
                    best_metrics = combined_metrics
                    best_cb_vectors = cb_vec.copy()
                    # Lưu model tạm thời
                    cb_sim_file = PHASE0_DIR / "cb_similarity_best.npz"
                    save_npz(cb_sim_file, cb_sim)

                del cb_sim; gc.collect()

    if best_params:
        save_best(PHASE0_DIR, best_params, best_metrics, PHASE0_DIR / "cb_similarity_best.npz")
        # Copy best model về models/ để các phase sau dùng
        import shutil
        shutil.copy(PHASE0_DIR / "cb_similarity_best.npz", MODELS_DIR / "cb_similarity.npz")
        print(f"\n  [Tune] Phase 0 done. Best: {best_params}")
        return best_cb_vectors
    else:
        print("\n  [Tune] Phase 0: No valid CB params found!")
        return {}


# ============================================================
# PHASE 1: SPMI (Collaborative Filtering)
# ============================================================

def tune_spmi(prior_df, train_df):
    """
    Grid search SPMI params:
      SPMI_K ∈ [3, 5, 7, 10, 15, 20]
      SPMI_TOP_K ∈ [50, 100, 200]
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Tuning SPMI")
    print("=" * 60)

    grid_k = [3, 5, 7, 10, 15, 20]
    grid_topk = [50, 100, 200]

    # Build co-occurrence chỉ 1 lần (tốn thời gian nhất)
    from src.features.build_spmi import build_cooc
    print("\n  [Tune] Building co-occurrence matrix (once)...")
    cooc, freq = build_cooc(prior_df)

    best_metric_val = -1.0
    best_params = None
    best_metrics = None

    total = len(grid_k) * len(grid_topk)
    current = 0

    for k in grid_k:
        for topk in grid_topk:
            current += 1
            params = {
                "SPMI_K": k,
                "SPMI_TOP_K": topk,
            }
            print(f"\n  --- [{current}/{total}] SPMI: k={k}, top_k={topk} ---")
            t0 = time.time()

            # Override config
            import src.config as cfg
            cfg.SPMI_K = k
            cfg.SPMI_TOP_K = topk

            # Build SPMI matrix
            from src.features.build_spmi import build_spmi
            spmi = build_spmi(cooc, freq, k=k, top_k=topk)

            # Evaluate
            metrics = evaluate_model(spmi, train_df, name=f"SPMI(k={k})", ks=EVAL_KS)
            elapsed = time.time() - t0

            combined_metrics = {"SPMI": metrics}
            save_snapshot(PHASE1_DIR, "spmi", params, combined_metrics, elapsed)

            metric_val = extract_metric(metrics)
            if metric_val > best_metric_val:
                best_metric_val = metric_val
                best_params = params.copy()
                best_metrics = combined_metrics
                spmi_file = PHASE1_DIR / "spmi_matrix_best.npz"
                save_npz(spmi_file, spmi)

            del spmi; gc.collect()

    del cooc, freq; gc.collect()

    if best_params:
        save_best(PHASE1_DIR, best_params, best_metrics, PHASE1_DIR / "spmi_matrix_best.npz")
        import shutil
        shutil.copy(PHASE1_DIR / "spmi_matrix_best.npz", MODELS_DIR / "spmi_matrix.npz")
        print(f"\n  [Tune] Phase 1 done. Best: {best_params}")
    else:
        print("\n  [Tune] Phase 1: No valid SPMI params found!")

    return load_npz(MODELS_DIR / "spmi_matrix.npz")


# ============================================================
# PHASE 2: Knowledge Graph (KG)
# ============================================================

def tune_kg(spmi, products_df, prior_df, train_df):
    """
    Grid search KG params:
      KG_DIM ∈ [32, 64]
      KG_WALK_LENGTH ∈ [10, 20, 30]
      KG_NUM_WALKS ∈ [30, 50]
      KG_EPOCHS ∈ [1, 3]
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Tuning Knowledge Graph (KG)")
    print("=" * 60)

    grid_dim = [32, 64]
    grid_walk = [10, 20, 30]
    grid_nwalks = [30, 50]
    grid_epochs = [1, 3]

    # Build graph chỉ 1 lần (dùng SPMI best)
    print("\n  [Tune] Building graph (once)...")
    from src.features.build_knowledge_graph import build_graph
    G = build_graph(spmi, products_df)

    best_metric_val = -1.0
    best_params = None
    best_metrics = None

    total = len(grid_dim) * len(grid_walk) * len(grid_nwalks) * len(grid_epochs)
    current = 0

    for dim in grid_dim:
        for walk_len in grid_walk:
            for n_walks in grid_nwalks:
                for epochs in grid_epochs:
                    current += 1
                    params = {
                        "KG_DIM": dim,
                        "KG_WALK_LENGTH": walk_len,
                        "KG_NUM_WALKS": n_walks,
                        "KG_EPOCHS": epochs,
                    }
                    print(f"\n  --- [{current}/{total}] KG: dim={dim}, walk={walk_len}, "
                          f"n_walks={n_walks}, epochs={epochs} ---")
                    t0 = time.time()

                    # Override config
                    import src.config as cfg
                    cfg.KG_DIM = dim
                    cfg.KG_WALK_LENGTH = walk_len
                    cfg.KG_NUM_WALKS = n_walks
                    cfg.KG_EPOCHS = epochs

                    # Train node2vec
                    from src.features.build_knowledge_graph import train_node2vec, compute_similarity
                    n_products = max(spmi.shape[0], int(prior_df["product_id"].max()) + 1)
                    emb = train_node2vec(G, n_products, dim=dim, walk_len=walk_len,
                                         n_walks=n_walks, epochs=epochs)
                    # Compute similarity
                    kg_sim = compute_similarity(emb, top_k=100)
                    del emb; gc.collect()

                    # Evaluate
                    metrics = evaluate_model(kg_sim, train_df, name=f"KG(dim={dim})", ks=EVAL_KS)
                    elapsed = time.time() - t0

                    combined_metrics = {"KG": metrics}
                    save_snapshot(PHASE2_DIR, "kg", params, combined_metrics, elapsed)

                    metric_val = extract_metric(metrics)
                    if metric_val > best_metric_val:
                        best_metric_val = metric_val
                        best_params = params.copy()
                        best_metrics = combined_metrics
                        kg_file = PHASE2_DIR / "kg_similarity_best.npz"
                        save_npz(kg_file, kg_sim)

                    del kg_sim; gc.collect()

    del G; gc.collect()

    if best_params:
        save_best(PHASE2_DIR, best_params, best_metrics, PHASE2_DIR / "kg_similarity_best.npz")
        import shutil
        shutil.copy(PHASE2_DIR / "kg_similarity_best.npz", MODELS_DIR / "kg_similarity.npz")
        print(f"\n  [Tune] Phase 2 done. Best: {best_params}")
    else:
        print("\n  [Tune] Phase 2: No valid KG params found!")

    return load_npz(MODELS_DIR / "kg_similarity.npz")


# ============================================================
# PHASE 3: Hybrid
# ============================================================

def tune_hybrid(spmi, kg_sim, cb_vectors, train_df):
    """
    Grid search Hybrid params:
      HYBRID_ALPHA ∈ [0.0, 0.1, 0.2, 0.3, 0.5]
      HYBRID_BETA ∈ [0.5, 0.7, 0.8, 0.9, 1.0]
      HYBRID_CB_THRESH ∈ [0.7, 0.8, 0.85, 0.9, 1.0]
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: Tuning Hybrid")
    print("=" * 60)

    grid_alpha = [0.0, 0.1, 0.2, 0.3, 0.5]
    grid_beta = [0.5, 0.7, 0.8, 0.9, 1.0]
    grid_cb_thresh = [0.7, 0.8, 0.85, 0.9, 1.0]

    best_metric_val = -1.0
    best_params = None
    best_metrics = None

    total = len(grid_alpha) * len(grid_beta) * len(grid_cb_thresh)
    current = 0

    for alpha in grid_alpha:
        for beta in grid_beta:
            # Bỏ cặp alpha+beta đều = 0 (không có nghĩa)
            if alpha == 0.0 and beta == 0.0:
                continue
            for cb_thresh in grid_cb_thresh:
                current += 1
                params = {
                    "HYBRID_ALPHA": alpha,
                    "HYBRID_BETA": beta,
                    "HYBRID_CB_THRESH": cb_thresh,
                }
                print(f"\n  --- [{current}/{total}] Hybrid: α={alpha}, β={beta}, cb_thresh={cb_thresh} ---")
                t0 = time.time()

                # Override config
                import src.config as cfg
                cfg.HYBRID_ALPHA = alpha
                cfg.HYBRID_BETA = beta
                cfg.HYBRID_CB_THRESH = cb_thresh

                # Build Hybrid matrix
                from src.features.build_hybrid import build_hybrid
                hybrid = build_hybrid(spmi, kg_sim, cb_vectors,
                                      alpha=alpha, beta=beta, cb_thresh=cb_thresh)

                # Evaluate (chỉ Hybrid)
                metrics = evaluate_model(hybrid, train_df, name=f"Hybrid(α={alpha})", ks=EVAL_KS)
                elapsed = time.time() - t0

                combined_metrics = {"Hybrid": metrics}
                save_snapshot(PHASE3_DIR, "hybrid", params, combined_metrics, elapsed)

                metric_val = extract_metric(metrics)
                if metric_val > best_metric_val:
                    best_metric_val = metric_val
                    best_params = params.copy()
                    best_metrics = combined_metrics
                    hybrid_file = PHASE3_DIR / "hybrid_matrix_best.npz"
                    save_npz(hybrid_file, hybrid)

                del hybrid; gc.collect()

    if best_params:
        save_best(PHASE3_DIR, best_params, best_metrics, PHASE3_DIR / "hybrid_matrix_best.npz")
        import shutil
        shutil.copy(PHASE3_DIR / "hybrid_matrix_best.npz", MODELS_DIR / "hybrid_matrix.npz")
        print(f"\n  [Tune] Phase 3 done. Best: {best_params}")
    else:
        print("\n  [Tune] Phase 3: No valid Hybrid params found!")


# ============================================================
# Tổng hợp final best params
# ============================================================

def save_final_best():
    """Gộp best params từ tất cả phases vào 1 file."""
    final = {}
    for phase_dir, phase_name in [
        (PHASE0_DIR, "CB"),
        (PHASE1_DIR, "SPMI"),
        (PHASE2_DIR, "KG"),
        (PHASE3_DIR, "Hybrid"),
    ]:
        bp_file = phase_dir / "best_params.json"
        if bp_file.exists():
            with open(bp_file, "r", encoding="utf-8") as f:
                final[phase_name] = json.load(f)
        else:
            final[phase_name] = None

    final_file = TUNE_DIR / "final_best_params.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n  [Tune] Final best params saved: {final_file}")


# ============================================================
# Main
# ============================================================

def main():
    """Chạy toàn bộ grid search 4 phase."""
    total_t0 = time.time()

    print("=" * 60)
    print("  HYPERPARAMETER TUNING — GRID SEARCH")
    print("=" * 60)
    print(f"  Best metric: {BEST_METRIC}")
    print(f"  Output: {TUNE_DIR}")

    # Load dữ liệu
    print("\n  [Tune] Loading data ...")
    products_df = load_products()
    prior_df = load_prior()
    train_df, _ = load_train_test()
    print(f"  [Tune] Products: {len(products_df):,}, Prior: {len(prior_df):,}, "
          f"Train orders: {len(train_df['order_id'].unique()):,}")

    # === Phase 0: CB ===
    cb_vec = tune_cb(products_df, prior_df, train_df)

    # === Phase 1: SPMI ===
    spmi = tune_spmi(prior_df, train_df)

    # === Phase 2: KG ===
    kg_sim = tune_kg(spmi, products_df, prior_df, train_df)

    # === Phase 3: Hybrid ===
    tune_hybrid(spmi, kg_sim, cb_vec, train_df)

    # === Final ===
    save_final_best()

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  TUNING COMPLETE! Total time: {total_elapsed / 3600:.2f} hours "
          f"({total_elapsed:.0f} seconds)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()