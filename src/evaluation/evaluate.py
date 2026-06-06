"""
Đánh giá các model trên tập TEST
Dùng leave-one-out: mỗi sản phẩm trong đơn → recommend top-K → hit?
"""
import gc, json
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm

from src.config import MODELS_DIR, RESULTS_DIR, EVAL_KS

METRICS_FILE = RESULTS_DIR / "metrics.json"

def evaluate_model(matrix, test_df, name="model", ks=EVAL_KS):
    """
    Đánh giá recall@k trên test set.
    matrix: csr_matrix (n×n) — similarity/score matrix
    test_df: DataFrame (order_id, product_id)
    """
    print(f"\n  [Eval] Evaluating {name} on test ...")
    groups = test_df.groupby("order_id")["product_id"].apply(list)
    hits = {k: 0 for k in ks}
    total = 0

    for prods in tqdm(groups.values, desc=f"  {name}"):
        if len(prods) < 2:
            continue
        prods_set = set(prods)
        for i, query in enumerate(prods):
            gt = prods_set - {query}
            row = matrix[query]
            if row.nnz == 0:
                continue
            # Sắp xếp theo score giảm dần
            order = np.argsort(row.data)[::-1]
            top = row.indices[order[:max(ks)]]
            for k in ks:
                if set(top[:k]) & gt:
                    hits[k] += 1
            total += 1

    metrics = {}
    for k in ks:
        recall = hits[k] / total if total > 0 else 0
        metrics[f"recall@{k}"] = round(recall, 6)
        print(f"    recall@{k} = {recall:.4f}")
    return metrics

def evaluate_all(test_df):
    """Đánh giá tất cả models trên test, lưu metrics"""
    results = {}

    # CB — chỉ evaluate để so sánh, không dùng để recommend
    try:
        import json as _json
        from src.features.build_cb import load as load_cb, cb_vectors
        # CB không có ma trận similarity dạng sparse, tính on-the-fly
        # Thay vào đó dùng spmi để test
    except:
        pass

    # SPMI
    try:
        spmi = load_npz(MODELS_DIR / "spmi_matrix.npz")
        results["SPMI"] = evaluate_model(spmi, test_df, "SPMI")
        del spmi; gc.collect()
    except Exception as e:
        print(f"  Skip SPMI: {e}")

    # KG
    try:
        kg = load_npz(MODELS_DIR / "kg_similarity.npz")
        results["KG"] = evaluate_model(kg, test_df, "KG")
        del kg; gc.collect()
    except Exception as e:
        print(f"  Skip KG: {e}")

    # Hybrid
    try:
        hybrid = load_npz(MODELS_DIR / "hybrid_matrix.npz")
        results["Hybrid"] = evaluate_model(hybrid, test_df, "Hybrid")
        del hybrid; gc.collect()
    except Exception as e:
        print(f"  Skip Hybrid: {e}")

    # Lưu
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [Eval] Saved: {METRICS_FILE}")
    return results

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(MODELS_DIR.parent))
    from src.data_loader import load_train_test
    _, test_df = load_train_test()
    evaluate_all(test_df)