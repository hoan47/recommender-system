"""
Đánh giá các model trên tập TEST bằng giao thức leave-one-out

Với mỗi sản phẩm trong mỗi đơn hàng (của test set):
  - Query = sản phẩm đó
  - Ground truth = các sản phẩm còn lại trong cùng đơn hàng
  - Lấy top-K recommendations từ similarity/score matrix
  - Hit = có ít nhất 1 ground truth trong top-K

Đánh giá trên SPMI, KG, và Hybrid (nếu có file).
Kết quả lưu vào results/metrics.json.
"""

import gc
import json
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm

from src.config import MODELS_DIR, RESULTS_DIR, EVAL_KS

# File lưu kết quả đánh giá
METRICS_FILE = RESULTS_DIR / "metrics.json"

def evaluate_model(matrix, test_df, name="model", ks=EVAL_KS):
    """
    Đánh giá recall@K trên test set.
    
    Tham số:
        matrix: csr_matrix (n_products x n_products) — ma trận similarity/score
        test_df: DataFrame (order_id, product_id) — ground truth test
        name: str — tên model (để hiển thị)
        ks: tuple — các giá trị K cần đánh giá
    
    Trả về:
        dict: {f"recall@{k}": value} cho mỗi k
    """
    print(f"\n  [Eval] Evaluating {name} on test ...")
    groups = test_df.groupby("order_id")["product_id"].apply(list)
    hits = {k: 0 for k in ks}
    total = 0

    for prods in tqdm(groups.values, desc=f"  {name}"):
        if len(prods) < 2:
            continue  # Bỏ đơn 1 sản phẩm (không có ground truth)
        prods_set = set(prods)
        for query in prods:
            gt = prods_set - {query}  # Các sản phẩm còn lại trong đơn
            row = matrix[query]
            if row.nnz == 0:
                continue
            # Sắp xếp recommendations theo score giảm dần
            order = np.argsort(row.data)[::-1]
            top = row.indices[order[:max(ks)]]  # Lấy top-k lớn nhất
            for k in ks:
                if set(top[:k]) & gt:  # Có ít nhất 1 ground truth trong top-K?
                    hits[k] += 1
            total += 1

    metrics = {}
    for k in ks:
        recall = hits[k] / total if total > 0 else 0
        metrics[f"recall@{k}"] = round(recall, 6)
        print(f"    recall@{k} = {recall:.4f}")
    return metrics

def evaluate_all(test_df):
    """
    Đánh giá tất cả models có sẵn trên test set.
    Tải ma trận từ models/ và chạy evaluation.
    """
    results = {}

    # Đánh giá SPMI
    try:
        spmi = load_npz(MODELS_DIR / "spmi_matrix.npz")
        results["SPMI"] = evaluate_model(spmi, test_df, "SPMI")
        del spmi; gc.collect()
    except Exception as e:
        print(f"  Skip SPMI: {e}")

    # Đánh giá KG
    try:
        kg = load_npz(MODELS_DIR / "kg_similarity.npz")
        results["KG"] = evaluate_model(kg, test_df, "KG")
        del kg; gc.collect()
    except Exception as e:
        print(f"  Skip KG: {e}")

    # Đánh giá Hybrid
    try:
        hybrid = load_npz(MODELS_DIR / "hybrid_matrix.npz")
        results["Hybrid"] = evaluate_model(hybrid, test_df, "Hybrid")
        del hybrid; gc.collect()
    except Exception as e:
        print(f"  Skip Hybrid: {e}")

    # Lưu kết quả ra file
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