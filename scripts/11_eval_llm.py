"""
11 — Đánh giá các model gợi ý mua kèm dùng ground truth từ LLM (Gemini).

Cách hoạt động:
1. Load gemini_responses_filtered.csv làm ground truth (các cặp complementary = label 1)
2. Load từng model đã train (Item-CF, Item2Vec, KGMetapath, Ensemble, Ensemble+CB)
3. Với mỗi product_A_id có trong ground truth:
   - Gọi model.recommend(product_A_id, top_k=10)
   - Đối chiếu top-10 với ground truth
   - Tính Precision@10, Recall@10, F1@10, Hit@10
4. Tính trung bình các metrics cho mỗi model
5. In bảng so sánh và lưu kết quả

Cách dùng:
   python scripts/11_eval_llm.py

Yêu cầu:
   - Đã train xong tất cả model (scripts/03 → 07)
   - File data/survey/llm_raw_responses/gemini_responses_filtered.csv tồn tại
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR, RESULT_DIR


# ============================================================
# Đường dẫn ground truth
# ============================================================
GT_PATH = os.path.join(
    MODEL_DIR, "..", "data", "survey",
    "llm_raw_responses", "gemini_responses_filtered.csv"
)


# ============================================================
# 1. Load ground truth
# ============================================================
def load_ground_truth():
    """
    Load gemini_responses_filtered.csv — chỉ chứa các cặp complementary.
    Trả về dict: product_A_id → set of product_B_id
    """
    df = pd.read_csv(GT_PATH, encoding='utf-8')
    
    # Tạo mapping: product_A_id → set of product_B_id (complementary)
    gt = {}
    for _, row in df.iterrows():
        pid_a = int(row['product_A_id'])
        pid_b = int(row['product_B_id'])
        if pid_a not in gt:
            gt[pid_a] = set()
        gt[pid_a].add(pid_b)
    
    print(f"  Ground truth: {len(gt)} target products, {len(df)} complementary pairs")
    return gt


# ============================================================
# 2. Load models
# ============================================================
def load_item_cf():
    """Load Item-CF model."""
    from src.models.item_cf import ItemCFModel
    model = ItemCFModel()
    model.load(os.path.join(MODEL_DIR, "item_cf"))
    return model


def load_item2vec():
    """Load Item2Vec (Neural CF) model."""
    from src.models.item_cf_neural import ItemCFNeuralModel
    model = ItemCFNeuralModel()
    model.load(os.path.join(MODEL_DIR, "item2vec"))
    return model


def load_kg_metapath():
    """Load KGMetapath model."""
    from src.models.kg_metapath import KGMetapathModel
    model = KGMetapathModel()
    model.load(os.path.join(MODEL_DIR, "kg_metapath"))
    return model


def load_ensemble(load_sub_models=True):
    """Load Ensemble model (có CB Filter bên trong)."""
    from src.models.ensemble import EnsembleModel
    ensemble = EnsembleModel.load(load_sub_models=load_sub_models)
    return ensemble


# ============================================================
# 3. Đánh giá metrics cho 1 model
# ============================================================
def evaluate_model(model, model_name, gt, top_k=10, valid_product_ids=None, **recommend_kwargs):
    """
    Đánh giá model dựa trên ground truth.
    
    Args:
        model: model object (phải có method recommend(product_id, top_k=...))
        model_name: str (dùng cho in ấn)
        gt: dict {product_A_id: set of product_B_id}
        top_k: int (default 10)
        valid_product_ids: set[int] — các product_id mà model biết (nếu None thì bỏ qua filter)
        **recommend_kwargs: các kwargs thêm cho recommend (vd use_cb_filter=True)
    
    Returns:
        dict: {model_name: {precision, recall, f1, hit}}
    """
    precisions = []
    recalls = []
    f1s = []
    hits = []
    n_skipped = 0
    n_evaluated = 0
    
    # Đảm bảo top_k công bằng:
    # - Ensemble: set cả top_k (số candidate từ mỗi sub-model) và final_k
    # - Các model khác: set top_k
    if hasattr(model, 'final_k'):
        orig_final_k = model.final_k
        model.final_k = top_k
    if hasattr(model, 'top_k'):
        orig_top_k = model.top_k
        model.top_k = top_k
    
    for pid_a, true_set in gt.items():
        # Pre-filter: bỏ qua nếu model không biết sản phẩm này
        if valid_product_ids is not None and pid_a not in valid_product_ids:
            n_skipped += 1
            continue
        
        try:
            if 'use_cb_filter' in recommend_kwargs:
                recs = model.recommend(pid_a, use_cb_filter=recommend_kwargs['use_cb_filter'])
            else:
                recs = model.recommend(pid_a, top_k=top_k)
        except KeyError:
            # Product không tồn tại trong model vocabulary
            n_skipped += 1
            continue
        except Exception as e:
            print(f"    ⚠️ Lỗi khi recommend product {pid_a}: {e}")
            n_skipped += 1
            continue
        
        if not recs:
            n_skipped += 1
            continue
        
        # Lấy danh sách product_id từ recommendations (bỏ qua score)
        n_recs = min(len(recs), top_k)
        pred_ids = [pid for pid, _ in recs[:n_recs]]
        
        # Đếm số lượng đúng
        n_correct = sum(1 for pid in pred_ids if pid in true_set)
        n_true = len(true_set)
        
        if n_true == 0:
            continue  # không có ground truth cho product này, bỏ qua
        
        # Precision@K chuẩn: luôn chia cho K
        # Nếu model trả về ít hơn K items, các vị trí thiếu được coi là không relevant → precision giảm
        precision = n_correct / top_k
        recall = n_correct / n_true
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        hit = 1 if n_correct > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        hits.append(hit)
        n_evaluated += 1
    
    # Khôi phục giá trị gốc
    if hasattr(model, 'final_k'):
        model.final_k = orig_final_k
    if hasattr(model, 'top_k'):
        model.top_k = orig_top_k
    
    # Tính trung bình
    result = {
        'model': model_name,
        'n_targets': len(gt),
        'n_evaluated': n_evaluated,
        'n_skipped': n_skipped,
        'precision@10': np.mean(precisions) if precisions else 0.0,
        'recall@10': np.mean(recalls) if recalls else 0.0,
        'f1@10': np.mean(f1s) if f1s else 0.0,
        'hit@10': np.mean(hits) if hits else 0.0,
    }
    return result


def print_results(results):
    """In bảng so sánh kết quả."""
    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ MODEL — GROUND TRUTH TỪ LLM (Gemini)")
    print("=" * 80)
    
    header = f"{'Model':<25} {'Targets':>8} {'Eval':>8} {'Skip':>8} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'Hit@10':>8}"
    print(header)
    print("-" * 80)
    
    for r in results:
        line = f"{r['model']:<25} {r['n_targets']:>8} {r['n_evaluated']:>8} {r['n_skipped']:>8} {r['precision@10']:>8.4f} {r['recall@10']:>8.4f} {r['f1@10']:>8.4f} {r['hit@10']:>8.4f}"
        print(line)
    
    print("=" * 80)


# ============================================================
# 4. Main
# ============================================================
def main():
    print("=" * 60)
    print("SCRIPT 11: ĐÁNH GIÁ MODEL BẰNG LLM GROUND TRUTH")
    print("=" * 60)
    
    # 1. Load ground truth
    print("\n📖 Đang load ground truth từ Gemini responses...")
    gt = load_ground_truth()
    
    # 2. Load models
    print("\n🧠 Đang load các model...")
    
    print("  Loading Item-CF...")
    item_cf = load_item_cf()
    
    print("  Loading Item2Vec...")
    item2vec = load_item2vec()
    
    print("  Loading KGMetapath...")
    kg_metapath = load_kg_metapath()
    
    print("  Loading Ensemble (w/ sub-models)...")
    ensemble = load_ensemble(load_sub_models=True)
    
    # Xác định valid product IDs cho từng model (pre-filter ground truth)
    item_cf_valid = set(item_cf.product_id_to_idx.keys())
    i2v_valid = set(int(k) for k in item2vec.model.wv.key_to_index.keys())
    mw_valid = set(kg_metapath.product_id_to_idx.keys())
    ensemble_valid = set(ensemble.item_cf.product_id_to_idx.keys())  # ensemble dùng chung product space
    
    # 3. Đánh giá từng model
    print("\n📊 Đang đánh giá các model...")
    results = []
    
    # Item-CF
    print("  Evaluating Item-CF...")
    r = evaluate_model(item_cf, "Item-CF", gt, valid_product_ids=item_cf_valid)
    results.append(r)
    
    # Item2Vec
    print("  Evaluating Item2Vec...")
    r = evaluate_model(item2vec, "Item2Vec", gt, valid_product_ids=i2v_valid)
    results.append(r)
    
    # KGMetapath
    print("  Evaluating KGMetapath...")
    r = evaluate_model(kg_metapath, "KGMetapath", gt, valid_product_ids=mw_valid)
    results.append(r)
    
    # Ensemble (w/o CB)
    print("  Evaluating Ensemble (w/o CB)...")
    r = evaluate_model(ensemble, "Ensemble (w/o CB)", gt, valid_product_ids=ensemble_valid, use_cb_filter=False)
    results.append(r)
    
    # Ensemble + CB
    print("  Evaluating Ensemble + CB...")
    r = evaluate_model(ensemble, "Ensemble + CB", gt, valid_product_ids=ensemble_valid, use_cb_filter=True)
    results.append(r)
    
    # 4. In kết quả
    print_results(results)
    
    # 5. Lưu kết quả
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, "llm_eval_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Đã lưu kết quả tại: {output_path}")


if __name__ == "__main__":
    main()