"""
Đánh giá các model trên tập TRAIN bằng giao thức leave-one-out

Với mỗi sản phẩm trong mỗi đơn hàng (của train set):
  - Query = sản phẩm đó
  - Ground truth = các sản phẩm còn lại trong cùng đơn hàng
  - Lấy top-K recommendations từ similarity/score matrix
  - Hit = có ít nhất 1 ground truth trong top-K

GHI CHÚ: Dataset Instacart public KHÔNG cung cấp ground truth cho test set (75K orders).
Do đó evaluation chạy trên train set (131K orders có ground truth).

Đánh giá trên SPMI, KG, và Hybrid (nếu có file).
Kết quả lưu vào results/metrics.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
        test_df: DataFrame (order_id, product_id) — ground truth
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
            # Query là product_id (1-based), matrix row index là 0-based
            # Nếu matrix shape < max product_id + 1, cần chuyển đổi
            if query >= matrix.shape[0]:
                continue
            row = matrix[query]
            if row.nnz == 0:
                continue
            # Sắp xếp recommendations theo score giảm dần
            order = np.argsort(row.data)[::-1]
            max_k = max(ks)
            top = row.indices[order[:max_k]]  # Lấy top-k lớn nhất
            for k in ks:
                if set(top[:k]) & gt:  # Có ít nhất 1 ground truth trong top-K?
                    hits[k] += 1
            total += 1

    metrics = {}
    for k in ks:
        recall = hits[k] / total if total > 0 else 0
        metrics[f"recall@{k}"] = round(recall, 6)
        print(f"    recall@{k} = {recall:.4f} (hits={hits[k]}, total={total})")
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
    # Dataset Instacart public KHÔNG có ground truth cho test set.
    # Chỉ train set (131K orders) có ground truth trong order_products__train.csv
    # Do đó dùng train set để đánh giá.
    train_df, _ = load_train_test()
    evaluate_all(train_df)