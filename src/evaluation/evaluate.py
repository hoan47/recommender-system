"""
Đánh giá các model trên tập TRAIN bằng giao thức leave-one-out

Với mỗi sản phẩm trong mỗi đơn hàng (của train set):
  - Query = sản phẩm đó
  - Ground truth = các sản phẩm còn lại trong cùng đơn hàng
  - Lấy top-K recommendations từ similarity/score matrix
  - Tính: Hit Rate@K, Precision@K, F1@K, NDCG@K, MAP@K

GHI CHÚ: Dataset Instacart public KHÔNG cung cấp ground truth cho test set (75K orders).
Do đó evaluation chạy trên train set (131K orders có ground truth).

Đánh giá trên CB, SPMI, KG, và Hybrid (nếu có file).
Kết quả lưu vào results/metrics.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import json
import math
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm

from src.config import MODELS_DIR, RESULTS_DIR, EVAL_KS

# File lưu kết quả đánh giá
METRICS_FILE = RESULTS_DIR / "metrics.json"


def ndcg_at_k(ranked_list, gt_set, k):
    """
    Tính NDCG@K.
    
    Tham số:
        ranked_list: list[int] — danh sách product IDs đã sort theo score giảm dần
        gt_set: set[int] — ground truth products
        k: int — K
    
    Trả về:
        float — NDCG@K (0.0 nếu không có ground truth)
    """
    if not gt_set:
        return 0.0
    
    dcg = 0.0
    ideal_dcg = 0.0
    n_rel = min(len(gt_set), k)
    
    for i in range(k):
        if i < len(ranked_list) and ranked_list[i] in gt_set:
            # gain = 1, discount = log2(i+2)
            dcg += 1.0 / math.log2(i + 2)
    
    # Ideal DCG: top min(K, |gt|) relevant items at ranks 0..n_rel-1
    for i in range(n_rel):
        ideal_dcg += 1.0 / math.log2(i + 2)
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def ap_at_k(ranked_list, gt_set, k):
    """
    Tính Average Precision @K.
    
    Tham số:
        ranked_list: list[int] — danh sách product IDs đã sort theo score giảm dần
        gt_set: set[int] — ground truth products
        k: int — K
    
    Trả về:
        float — AP@K (0.0 nếu không có ground truth)
    """
    if not gt_set:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    n_rel = min(len(gt_set), k)
    
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i] in gt_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    
    return sum_precisions / n_rel if n_rel > 0 else 0.0


def evaluate_model(matrix, test_df, name="model", ks=EVAL_KS):
    """
    Đánh giá Hit Rate@K, Precision@K, F1@K, NDCG@K, MAP@K trên test set.
    
    Tham số:
        matrix: csr_matrix (n_products x n_products) — ma trận similarity/score
        test_df: DataFrame (order_id, product_id) — ground truth
        name: str — tên model (để hiển thị)
        ks: tuple — các giá trị K cần đánh giá
    
    Trả về:
        dict: {metric@k: value} cho mỗi k
    """
    print(f"\n  [Eval] Evaluating {name} on test ...")
    groups = test_df.groupby("order_id")["product_id"].apply(list)
    
    # Khởi tạo accumulators cho từng metric
    recall_hits = {k: 0 for k in ks}       # Số query có ít nhất 1 hit (cho Hit Rate)
    precision_hits = {k: 0 for k in ks}    # Tổng số hits trong top-K (cho Precision)
    recall_total = 0
    ndcg_sums = {k: 0.0 for k in ks}
    ap_sums = {k: 0.0 for k in ks}
    query_count = 0

    for prods in tqdm(groups.values, desc=f"  {name}"):
        if len(prods) < 2:
            continue  # Bỏ đơn 1 sản phẩm (không có ground truth)
        prods_set = set(prods)
        for query in prods:
            gt = prods_set - {query}  # Các sản phẩm còn lại trong đơn
            if query >= matrix.shape[0]:
                continue
            row = matrix[query]
            if row.nnz == 0:
                continue
            
            # Sắp xếp recommendations theo score giảm dần
            order = np.argsort(row.data)[::-1]
            max_k = max(ks)
            # Lấy top-K lớn nhất, chuyển về list int
            top = row.indices[order[:max_k]].tolist()
            
            for k in ks:
                topk_set = set(top[:k])
                # Hit Rate: có ít nhất 1 ground truth trong top-K?
                if topk_set & gt:
                    recall_hits[k] += 1
                
                # Precision: đếm số lượng ground truth trong top-K
                precision_hits[k] += len(topk_set & gt)
                
                # NDCG@K
                ndcg_sums[k] += ndcg_at_k(top[:k], gt, k)
                
                # MAP@K
                ap_sums[k] += ap_at_k(top[:k], gt, k)
            
            recall_total += 1
            query_count += 1

    metrics = {}
    for k in ks:
        # Hit Rate@K (= Recall@K cũ): tỷ lệ query có ít nhất 1 ground truth trong top-K
        hit = recall_hits[k] / recall_total if recall_total > 0 else 0
        metrics[f"hit@{k}"] = round(hit, 6)
        
        # Precision@K: trung bình số lượng ground truth trong top-K, chia cho K
        precision = precision_hits[k] / (query_count * k) if query_count > 0 else 0
        metrics[f"precision@{k}"] = round(precision, 6)
        
        # F1@K: harmonic mean của Hit Rate và Precision
        if hit > 0 and precision > 0:
            f1 = 2 * hit * precision / (hit + precision)
        else:
            f1 = 0.0
        metrics[f"f1@{k}"] = round(f1, 6)
        
        # NDCG@K
        ndcg_avg = ndcg_sums[k] / query_count if query_count > 0 else 0
        metrics[f"ndcg@{k}"] = round(ndcg_avg, 6)
        
        # MAP@K
        ap_avg = ap_sums[k] / query_count if query_count > 0 else 0
        metrics[f"map@{k}"] = round(ap_avg, 6)
        
        print(f"    hit@{k} = {hit:.4f}  precision@{k} = {precision:.4f}  f1@{k} = {f1:.4f}  "
              f"ndcg@{k} = {ndcg_avg:.4f}  map@{k} = {ap_avg:.4f}  "
              f"(pra_hits={precision_hits[k]}, queries={query_count})")
    
    return metrics


def evaluate_all(test_df):
    """
    Đánh giá tất cả models có sẵn trên test set.
    Tải ma trận từ models/ và chạy evaluation.
    """
    results = {}

    # # Đánh giá CB (standalone)
    # try:
    #     cb = load_npz(MODELS_DIR / "cb_similarity.npz")
    #     results["CB"] = evaluate_model(cb, test_df, "CB")
    #     del cb; gc.collect()
    # except Exception as e:
    #     print(f"  Skip CB: {e}")

    # Đánh giá SPMI
    try:
        spmi = load_npz(MODELS_DIR / "spmi_matrix.npz")
        results["SPMI"] = evaluate_model(spmi, test_df, "SPMI")
        del spmi; gc.collect()
    except Exception as e:
        print(f"  Skip SPMI: {e}")

    # # Đánh giá KG
    # try:
    #     kg = load_npz(MODELS_DIR / "kg_similarity.npz")
    #     results["KG"] = evaluate_model(kg, test_df, "KG")
    #     del kg; gc.collect()
    # except Exception as e:
    #     print(f"  Skip KG: {e}")

    # # Đánh giá Hybrid
    # try:
    #     hybrid = load_npz(MODELS_DIR / "hybrid_matrix.npz")
    #     results["Hybrid"] = evaluate_model(hybrid, test_df, "Hybrid")
    #     del hybrid; gc.collect()
    # except Exception as e:
    #     print(f"  Skip Hybrid: {e}")

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