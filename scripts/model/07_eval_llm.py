import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import MODEL_DIR, RESULT_DIR

GT_PATH = os.path.join(
    MODEL_DIR, "..", "data", "survey",
    "llm_raw_responses", "gemini_responses_filtered.csv"
)

# ============================================================
# 1. Load ground truth
# ============================================================
def load_ground_truth():
    df = pd.read_csv(GT_PATH, encoding='utf-8')
    gt = {}
    for _, row in df.iterrows():
        pid_a = int(row['product_A_id'])
        pid_b = int(row['product_B_id'])
        if pid_a not in gt:
            gt[pid_a] = set()
        gt[pid_a].add(pid_b)
    return gt

# ============================================================
# 2. Load models (Giữ nguyên như code của bạn)
# ============================================================
def load_item_cf():
    from src.models.item_cf import ItemCFModel
    model = ItemCFModel()
    model.load(os.path.join(MODEL_DIR, "item_cf"))
    return model

def load_item2vec():
    from src.models.item_cf_neural import ItemCFNeuralModel
    model = ItemCFNeuralModel()
    model.load(os.path.join(MODEL_DIR, "item2vec"))
    return model

def load_kg_metapath():
    from src.models.kg_metapath import KGMetapathModel
    model = KGMetapathModel()
    model.load(os.path.join(MODEL_DIR, "kg_metapath"))
    return model

def load_ensemble(load_sub_models=True):
    from src.models.ensemble import EnsembleModel
    return EnsembleModel.load(load_sub_models=load_sub_models)

# ============================================================
# 3. Đánh giá metrics cho 1 model (ĐÃ SỬA: KHÔNG SKIP BẤT KỲ TARGET NÀO)
# ============================================================
def evaluate_model(model, model_name, filtered_gt, top_k=10, **recommend_kwargs):
    precisions = []
    recalls = []
    f1s = []
    hits = []
    n_errors = 0  # Đếm số lượt model bị lỗi kỹ thuật hoặc crash khi gọi recommend
    
    if hasattr(model, 'final_k'):
        orig_final_k = model.final_k
        model.final_k = top_k
    if hasattr(model, 'top_k'):
        orig_top_k = model.top_k
        model.top_k = top_k
    
    # Chạy qua TẬP DỮ LIỆU DUY NHẤT đã được lọc chung
    for pid_a, true_set in filtered_gt.items():
        n_true = len(true_set)
        pred_ids = []
        
        try:
            # FIX LỖI: Luôn truyền top_k vào để đồng bộ số lượng item gợi ý
            if 'use_cb_filter' in recommend_kwargs:
                recs = model.recommend(pid_a, top_k=top_k, use_cb_filter=recommend_kwargs['use_cb_filter'])
            else:
                recs = model.recommend(pid_a, top_k=top_k)
                
            if recs:
                n_recs = min(len(recs), top_k)
                pred_ids = [pid for pid, _ in recs[:n_recs]]
        except Exception as e:
            # Nếu model không biết sản phẩm hoặc lỗi, pred_ids giữ nguyên là [] 
            # Điểm lượt này sẽ bằng 0 (Hoàn toàn công bằng vì đã lọc các ID chung từ trước)
            n_errors += 1
        
        # Tính toán metrics
        n_correct = sum(1 for pid in pred_ids if pid in true_set)
        
        precision = n_correct / top_k
        recall = n_correct / n_true
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        hit = 1 if n_correct > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        hits.append(hit)
    
    if hasattr(model, 'final_k'):
        model.final_k = orig_final_k
    if hasattr(model, 'top_k'):
        model.top_k = orig_top_k
    
    return {
        'model': model_name,
        'test_set_size': len(filtered_gt),
        'n_errors': n_errors, 
        'precision@10': np.mean(precisions) if precisions else 0.0,
        'recall@10': np.mean(recalls) if recalls else 0.0,
        'f1@10': np.mean(f1s) if f1s else 0.0,
        'hit@10': np.mean(hits) if hits else 0.0,
    }

def print_results(results):
    print("\n" + "=" * 90)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN MỘT TẬP DỮ LIỆU ĐỒNG NHẤT (FAIR BENCHMARK)")
    print("=" * 90)
    header = f"{'Model':<25} {'Test Size':>10} {'Errors':>8} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'Hit@10':>8}"
    print(header)
    print("-" * 90)
    for r in results:
        line = f"{r['model']:<25} {r['test_set_size']:>10} {r['n_errors']:>8} {r['precision@10']:>8.4f} {r['recall@10']:>8.4f} {r['f1@10']:>8.4f} {r['hit@10']:>8.4f}"
        print(line)
    print("=" * 90)

# ============================================================
# 4. Main
# ============================================================
def main():
    print("=" * 60)
    print("BƯỚC 7: ĐÁNH GIÁ ĐỒNG NHẤT BẰNG TẬP KIỂM THỬ CHUNG")
    print("=" * 60)
    
    # Load ground truth gốc từ LLM
    raw_gt = load_ground_truth()
    print(f"  Ground truth gốc: {len(raw_gt)} target products")
    
    # Load các model
    item_cf = load_item_cf()
    item2vec = load_item2vec()
    kg_metapath = load_kg_metapath()
    ensemble = load_ensemble(load_sub_models=True)
    
    # Lấy tập sản phẩm (Vocabulary) của từng model
    item_cf_vocab = set(item_cf.product_id_to_idx.keys())
    i2v_vocab = set(int(k) for k in item2vec.model.wv.key_to_index.keys())
    kg_vocab = set(kg_metapath.product_id_to_idx.keys())
    ensemble_vocab = set(ensemble.item_cf.product_id_to_idx.keys())
    
    # BƯỚC QUAN TRỌNG: Tìm tập giao (Intersection) của cả 4 mô hình và tập Ground Truth
    # Đảm bảo sản phẩm kiểm thử phải được TẤT CẢ các bên nhìn thấy công bằng
    common_product_ids = (
        set(raw_gt.keys()) & 
        item_cf_vocab & 
        i2v_vocab & 
        kg_vocab & 
        ensemble_vocab
    )
    
    # Tạo ra tập dữ liệu kiểm thử DUY NHẤT
    filtered_gt = {pid: raw_gt[pid] for pid in common_product_ids if len(raw_gt[pid]) > 0}
    print(f"  🎯 Tập dữ liệu đánh giá chung duy nhất: {len(filtered_gt)} target products")
    
    if len(filtered_gt) == 0:
        print("❌ Lỗi: Không có product_A_id chung nào giữa các model và Ground Truth! Vui lòng kiểm tra lại data train.")
        return

    # Tiến hành đánh giá đồng loạt trên tập dữ liệu chung này
    results = []
    
    results.append(evaluate_model(item_cf, "Item-CF", filtered_gt))
    results.append(evaluate_model(item2vec, "Item2Vec", filtered_gt))
    results.append(evaluate_model(kg_metapath, "KGMetapath", filtered_gt))
    results.append(evaluate_model(ensemble, "Ensemble (w/o CB)", filtered_gt, use_cb_filter=False))
    results.append(evaluate_model(ensemble, "Ensemble + CB", filtered_gt, use_cb_filter=True))
    
    # In và lưu kết quả
    print_results(results)
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, "llm_eval_results.csv")
    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Đã lưu kết quả đồng nhất tại: {output_path}")

if __name__ == "__main__":
    main()