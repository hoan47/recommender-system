"""
Bước 7: Đánh giá model bằng LLM (LLM Evaluation).
Chạy riêng: python scripts/model/07_eval_llm.py
Yêu cầu: scripts/model/01-06 đã chạy + gemini_responses_filtered.csv đã có
Output: results/llm_eval_results.csv (Precision, Recall, F1, Hit cho từng model)

Ground truth: data/survey/llm_raw_responses/gemini_responses_filtered.csv
  - 3 cột: product_A_id, product_B_id, description
  - Mỗi dòng là 1 cặp complementary (llm_label = 1)
  - Các cặp không có trong file này mặc định là not complementary (llm_label = 0)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.config import MODEL_DIR
from src.models.item_cf import ItemCFModel
from src.models.item_cf_neural import ItemCFNeuralModel
from src.models.kg_metapath import KGMetapathModel
from src.models.ensemble import EnsembleModel

print("="*60)
print("  BƯỚC 7: LLM EVALUATION")
print("="*60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
SURVEY_DIR = os.path.join(PROJECT_ROOT, "data", "survey")
GT_FILE = os.path.join(SURVEY_DIR, "llm_raw_responses", "gemini_responses_filtered.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Các model cần đánh giá
MODEL_NAMES = ['item_cf', 'item2vec', 'kg_metapath', 'ensemble', 'ensemble_cb']

TOP_K = 10


def load_ground_truth():
    """
    Load ground truth từ gemini_responses_filtered.csv.
    
    File này chỉ chứa các cặp complementary (label=1).
    Xây set comp_pairs = {(A_id, B_id), ...} để tra cứu nhanh.
    
    Returns:
        comp_pairs: set[(int, int)] — các cặp complementary
        all_targets: list[int] — các target product A duy nhất
    """
    df = pd.read_csv(GT_FILE)
    comp_pairs = set()
    for _, row in df.iterrows():
        comp_pairs.add((int(row['product_A_id']), int(row['product_B_id'])))
    
    all_targets = sorted(df['product_A_id'].unique())
    
    print(f"   -> {len(comp_pairs):,} complementary pairs")
    print(f"   -> {len(all_targets):,} unique target products")
    return comp_pairs, all_targets


def load_models():
    """Load các model cần đánh giá."""
    models = {}

    # Item-CF
    item_cf = ItemCFModel()
    item_cf.load(os.path.join(MODEL_DIR, "item_cf"))
    models['item_cf'] = item_cf

    # Item2Vec
    i2v = ItemCFNeuralModel()
    i2v.load(os.path.join(MODEL_DIR, "item2vec"))
    models['item2vec'] = i2v

    # KGMetapath
    mw = KGMetapathModel()
    mw.load(os.path.join(MODEL_DIR, "kg_metapath"))
    models['kg_metapath'] = mw

    # Ensemble
    ensemble = EnsembleModel.load(load_sub_models=False)
    ensemble.fit(item_cf, i2v, mw, ensemble.cb_filter)
    models['ensemble'] = ensemble

    return models


def evaluate_model(model, model_name, targets, comp_pairs, top_k=TOP_K):
    """
    Đánh giá một model trên tập target.
    
    Args:
        model: model object (có method recommend)
        model_name: str
        targets: list[int] — các product_id target
        comp_pairs: set[(int, int)] — ground truth complementary
        top_k: int
    
    Returns:
        dict: precision, recall, f1, hit
    """
    all_precisions = []
    all_recalls = []
    all_hits = []

    for pid_a in targets:
        # Lấy top-k gợi ý
        if model_name == 'ensemble_cb':
            recs = model.recommend(pid_a, use_cb_filter=True, top_k=top_k)
        else:
            recs = model.recommend(pid_a, top_k=top_k)

        pred_pids = [r[0] for r in recs[:top_k]]

        # Đếm số complementary trong top-k
        n_complementary = sum(
            1 for pid_b in pred_pids
            if (pid_a, pid_b) in comp_pairs
        )

        # Precision@k
        precision = n_complementary / top_k
        all_precisions.append(precision)

        # Hit@k
        hit = 1 if n_complementary > 0 else 0
        all_hits.append(hit)

        # Recall@k: số ground truth complementary cho pid_a
        n_gt = sum(1 for (a, b) in comp_pairs if a == pid_a)
        recall = n_complementary / n_gt if n_gt > 0 else 0
        all_recalls.append(recall)

    # Average
    avg_precision = np.mean(all_precisions) if all_precisions else 0
    avg_recall = np.mean(all_recalls) if all_recalls else 0
    avg_hit = np.mean(all_hits) if all_hits else 0
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return {
        'model': model_name,
        'precision': round(avg_precision, 4),
        'recall': round(avg_recall, 4),
        'f1': round(f1, 4),
        'hit': round(avg_hit, 4),
        'n_targets': len(targets),
    }


def main():
    print("\n1. Loading ground truth từ gemini_responses_filtered.csv...")
    comp_pairs, all_targets = load_ground_truth()

    print("\n2. Loading products...")
    products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
    print(f"   -> {len(products)} products")

    print("\n3. Loading models...")
    models = load_models()
    # Thêm ensemble_cb (dùng cùng ensemble model nhưng bật CB filter)
    models['ensemble_cb'] = models['ensemble']

    targets = all_targets
    print(f"\n4. Evaluating {len(targets)} targets (top-{TOP_K})...")

    results = []
    for model_name in MODEL_NAMES:
        print(f"\n   Evaluating {model_name}...")
        result = evaluate_model(
            models[model_name], model_name, targets, comp_pairs
        )
        results.append(result)
        print(f"     Precision@{TOP_K}: {result['precision']:.4f}")
        print(f"     Recall@{TOP_K}:    {result['recall']:.4f}")
        print(f"     F1@{TOP_K}:        {result['f1']:.4f}")
        print(f"     Hit@{TOP_K}:       {result['hit']:.4f}")

    # Lưu kết quả
    os.makedirs(RESULT_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULT_DIR, "llm_eval_results.csv"), index=False)
    print(f"\n✅ Results saved to {RESULT_DIR}llm_eval_results.csv")
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()