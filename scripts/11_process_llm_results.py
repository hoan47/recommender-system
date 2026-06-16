"""
11 — Process LLM raw responses -> labeled survey + model evaluation.

Pipeline:
  1. Doc survey_samples.csv (4 cot: A_id, A_name, B_id, B_name)
  2. Doc LLM raw responses tu data/survey/llm_raw_responses/
     (JSON array, moi object co target_id + recommendations{bid: description})
  3. Ghep: voi moi (A_id, B_id), gan llm_label=1 neu B trong recommendations, else 0
  4. Xuat survey_labeled.csv (6 cot)
  5. Tinh metrics (Precision@10, Recall@10, F1@10, Hit@10) cho tung model

Usage:
  python scripts/11_process_llm_results.py
"""
import os
import sys
import json
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_DIR

# ============================================================
# Cấu hình
# ============================================================
SURVEY_DIR = os.path.join(MODEL_DIR, "..", "data", "survey")
SURVEY_FILE = os.path.join(SURVEY_DIR, "survey_samples.csv")
LLM_RESPONSES_DIR = os.path.join(SURVEY_DIR, "llm_raw_responses")
LABELED_OUTPUT = os.path.join(SURVEY_DIR, "survey_labeled.csv")
STATS_OUTPUT = os.path.join(SURVEY_DIR, "survey_stats.txt")

# Danh sách model cần đánh giá (map ten -> model subfolder)
MODELS = {
    'item_cf': 'item_cf',
    'item2vec': 'item2vec',
    'kg_metapath': 'kg_metapath',
    'ensemble': 'ensemble',        # Ensemble w/o CB
    'ensemble_cb': 'ensemble',      # Ensemble + CB (CB la post-filter, cung model)
}

METRICS_OUTPUT = os.path.join(SURVEY_DIR, "model_metrics.csv")


def load_survey_samples():
    """Doc survey_samples.csv."""
    print("  Doc survey_samples.csv...")
    df = pd.read_csv(SURVEY_FILE, encoding='utf-8')
    print(f"    {len(df):,} dong, {df['product_A_id'].nunique():,} targets")
    return df


def load_llm_responses():
    """
    Doc tat ca JSON tu llm_raw_responses/.
    Tra ve dict: (target_id) -> dict{bid: description}
    """
    print("  Doc LLM raw responses...")
    all_responses = {}
    
    json_files = glob.glob(os.path.join(LLM_RESPONSES_DIR, "*.json"))
    if not json_files:
        print("    KHONG tim thay file JSON nao trong", LLM_RESPONSES_DIR)
        return all_responses
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support ca array va single object
        if isinstance(data, list):
            responses = data
        else:
            responses = [data]
        
        for resp in responses:
            target_id = resp.get('target_id')
            if not target_id:
                continue
            
            # recommendations co the la dict hoac list cac dict
            recs = resp.get('recommendations', {})
            if isinstance(recs, list):
                recs = {r.get('product_B_id', r.get('bid', '')): r.get('description', '')
                        for r in recs if r.get('product_B_id') or r.get('bid')}
            
            recommendations = {}
            for bid, description in recs.items():
                bid_str = str(bid).strip()
                if bid_str:
                    recommendations[bid_str] = str(description).strip()
            
            all_responses[target_id] = recommendations
    
    total_labeled = sum(len(recs) for recs in all_responses.values())
    print(f"    {len(all_responses)} targets duoc LLM danh gia")
    print(f"    {total_labeled:,} cap duoc gan nhan complementary (llm_label=1)")
    
    return all_responses


def create_labeled_survey(survey_df, llm_responses):
    """
    Ghep LLM responses voi survey -> survey_labeled.csv.
    """
    print("  Tao survey_labeled.csv...")
    
    # Tao dict lookup: (A_id, B_id) -> description
    label_map = {}  # (str(A_id), str(B_id)) -> description
    for target_id, recs in llm_responses.items():
        for bid_str, desc in recs.items():
            label_map[(str(target_id), bid_str)] = desc
    
    # Gan label cho tung dong
    rows = []
    n_labeled = 0
    for _, row in survey_df.iterrows():
        aid = str(row['product_A_id'])
        bid = str(row['product_B_id'])
        key = (aid, bid)
        
        if key in label_map:
            llm_label = 1
            description = label_map[key]
            n_labeled += 1
        else:
            llm_label = 0
            description = ''
        
        rows.append({
            'product_A_id': row['product_A_id'],
            'product_A_name': row['product_A_name'],
            'product_B_id': row['product_B_id'],
            'product_B_name': row['product_B_name'],
            'llm_label': llm_label,
            'description': description,
        })
    
    df_labeled = pd.DataFrame(rows)
    df_labeled.to_csv(LABELED_OUTPUT, index=False, encoding='utf-8')
    print(f"    {len(df_labeled):,} dong duoc ghi nhan")
    print(f"    Trong do {n_labeled:,} dong llm_label=1 ({n_labeled/len(df_labeled)*100:.1f}%)")
    
    return df_labeled


def load_model_topk(model_name, top_k=10):
    """
    Load top-K recommendations cua mot model tu file model output.
    Tra ve dict: product_A_id -> list[product_B_id]
    
    Note: Can custom theo format output cua tung model.
    Day la placeholder - can bo sung khi co file thuc te.
    """
    # Placeholder: doc tu survey samples (vi chua co file model topk rieng)
    # Trong thuc te, can doc tu model output files
    return {}


def evaluate_model(df_labeled, model_name, model_topk=None):
    """
    Tinh Precision@10, Recall@10, F1@10, Hit@10 cho mot model.
    
    Args:
        df_labeled: DataFrame voi cot llm_label, product_A_id, product_B_id
        model_name: Ten model de load topk (neu model_topk=None)
        model_topk: Dict pre-loaded {aid: [bid, ...]} (optional)
    
    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float, 'hit': float}
    """
    if model_topk is None:
        model_topk = load_model_topk(model_name)
    
    targets = df_labeled['product_A_id'].unique()
    
    # Ghep: voi moi target, xem top-10 cua model => precision
    # Can co top-10 thuc te cua model -> placeholder
    # Tinh toan cache:
    # TP = B trong top-10 AND llm_label=1
    # FP = B trong top-10 AND llm_label=0
    # FN = B KHONG trong top-10 AND llm_label=1
    # Precision@10 = TP / (TP + FP) = TP / 10
    # Recall@10 = TP / (TP + FN)
    # Hit@10 = 1 if TP > 0 else 0
    
    # Placeholder: tra ve 0
    return {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'hit': 0.0,
        'n_targets': 0,
    }


def calculate_metrics_from_llm(df_labeled, llm_responses):
    """
    Tinh metrics dua tren ground truth tu LLM.
    Dung llm_responses lam ground truth.
    
    Cach tinh:
      - Voi moi target A, lay danh sach B duoc LLM danh la complementary
      - Precision = |recs(A) inter LLM_complementary(A)| / len(recs(A))
      - Mac dinh recs(A) = tat ca candidate B tu survey (union 5 model)
    """
    print("  Tinh metrics tu LLM ground truth...")
    
    # Ground truth: target -> set cac B duoc LLM chon
    gt = {}
    for target_id, recs in llm_responses.items():
        gt[target_id] = set(recs.keys())
    
    # Model: tat ca candidate tu survey (union 5 model)
    model_recs = {}
    for _, row in df_labeled.iterrows():
        aid = str(row['product_A_id'])
        bid = str(row['product_B_id'])
        if aid not in model_recs:
            model_recs[aid] = []
        model_recs[aid].append(bid)
    
    # Chi tinh cho target co trong ca ground truth va model
    common_targets = set(gt.keys()) & set(model_recs.keys())
    
    precisions = []
    hits = []
    
    for tid in common_targets:
        recs = set(model_recs[tid])
        gt_set = gt[tid]
        
        # So luong complementary tu ground truth
        n_gt = len(gt_set)
        if n_gt == 0:
            continue
        
        # So luong B trong model_recs duoc xac nhan complementary
        n_correct = len(recs & gt_set)
        
        # Precision = correct / total_recs
        n_recs = len(recs)
        if n_recs > 0:
            precision = n_correct / n_recs
            precisions.append(precision)
            
            # Hit@K: co it nhat 1 dung?
            hits.append(1 if n_correct > 0 else 0)
    
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_hit = np.mean(hits) if hits else 0.0
    
    # Recall: trung binh ca target, recall = correct / min(n_gt, n_recs)
    recalls = []
    for tid in common_targets:
        recs = set(model_recs[tid])
        gt_set = gt[tid]
        n_gt = len(gt_set)
        if n_gt == 0:
            continue
        n_correct = len(recs & gt_set)
        # Recall = trong so complementary cua target, model gom duoc bao nhieu?
        recall = n_correct / n_gt
        recalls.append(recall)
    
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    # F1
    if avg_precision + avg_recall > 0:
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        avg_f1 = 0.0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'hit': avg_hit,
        'n_targets': len(common_targets),
        'n_gt_total': sum(len(gt[tid]) for tid in common_targets),
        'n_model_total': sum(len(model_recs[tid]) for tid in common_targets),
    }


def main():
    print("=" * 60)
    print("SCRIPT 11: PROCESS LLM RESULTS -> GROUND TRUTH + METRICS")
    print("=" * 60)
    
    # 1. Doc survey samples
    print("\n📋 Buoc 1: Doc survey samples")
    survey_df = load_survey_samples()
    
    # 2. Doc LLM responses
    print("\n🤖 Buoc 2: Doc LLM raw responses")
    llm_responses = load_llm_responses()
    
    if not llm_responses:
        print("\n⚠️  Khong co LLM responses. Thoat.")
        print("   Dat file JSON vao:", LLM_RESPONSES_DIR)
        print("   Format: [{'target_id':'123','recommendations':{'456':'Mo ta','789':'Mo ta'}},...]")
        return
    
    # 3. Tao survey_labeled.csv
    print("\n🏷️  Buoc 3: Tao ground truth (survey_labeled.csv)")
    df_labeled = create_labeled_survey(survey_df, llm_responses)
    
    # 4. Tinh metrics
    print("\n📊 Buoc 4: Tinh metrics")
    metrics = calculate_metrics_from_llm(df_labeled, llm_responses)
    
    print(f"\n   === KET QUA METRICS (Union 5 models) ===")
    print(f"   So target co ground truth: {metrics['n_targets']:,}")
    print(f"   Tong cap complementary (GT): {metrics['n_gt_total']:,}")
    print(f"   Tong cap model candidate: {metrics['n_model_total']:,}")
    print(f"   Precision (mac dinh):       {metrics['precision']:.4f}")
    print(f"   Recall (mac dinh):          {metrics['recall']:.4f}")
    print(f"   F1 (mac dinh):              {metrics['f1']:.4f}")
    print(f"   Hit@K (mac dinh):           {metrics['hit']:.4f}")
    print(f"\n   📌 LUU Y: day la metrics cho UNION cua 5 model.")
    print(f"   De tinh metrics cho tung model, can bo sung file top-K rieng.")
    
    # 5. Ghi stats
    print(f"\n📝 Buoc 5: Ghi thong ke")
    with open(STATS_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(f"Survey Statistics\n")
        f.write(f"{'='*40}\n")
        f.write(f"Survey samples:  {len(survey_df):,} dong\n")
        f.write(f"  - Targets:     {survey_df['product_A_id'].nunique():,}\n")
        f.write(f"  - Candidates:  {survey_df['product_B_id'].nunique():,}\n")
        f.write(f"\n")
        f.write(f"LLM responses:   {len(llm_responses)} targets\n")
        f.write(f"  - Tong cap llm_label=1: {sum(len(r) for r in llm_responses.values()):,}\n")
        f.write(f"\n")
        f.write(f"Labeled survey:  {len(df_labeled):,} dong\n")
        f.write(f"  - llm_label=1: {df_labeled['llm_label'].sum():,} (Tyle: {df_labeled['llm_label'].mean()*100:.1f}%)\n")
        f.write(f"  - llm_label=0: {len(df_labeled)-df_labeled['llm_label'].sum():,}\n")
        f.write(f"\n")
        f.write(f"Metrics (union 5 models):\n")
        f.write(f"  - Targets:     {metrics['n_targets']:,}\n")
        f.write(f"  - Precision:   {metrics['precision']:.4f}\n")
        f.write(f"  - Recall:      {metrics['recall']:.4f}\n")
        f.write(f"  - F1:          {metrics['f1']:.4f}\n")
        f.write(f"  - Hit@K:       {metrics['hit']:.4f}\n")
    
    print(f"   Da ghi: {STATS_OUTPUT}")
    print(f"\n✅ Hoan thanh!")
    print(f"\n📁 Cac file da tao:")
    print(f"   {LABELED_OUTPUT}")
    print(f"   {STATS_OUTPUT}")
    print(f"\n👉 Tiep theo: Chay script de tinh metrics cho tung model rieng biet")
    print(f"   (Can bo sung file top-K cua tung model truoc)")


if __name__ == "__main__":
    main()