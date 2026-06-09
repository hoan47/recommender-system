"""
Bước 8: Đánh giá Association Rules.
Sinh survey samples + tính metrics.
Chạy riêng: python scripts/08_evaluate_assoc_rules.py
Yêu cầu: scripts/01_load_data.py + scripts/06_assoc_rules.py đã chạy
Output: results/assoc_rules_evaluation/
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np

from src.config import MODEL_DIR, DATA_DIR, PROCESSED_DIR, RESULT_DIR, RANDOM_SEED
from src.models.assoc_rules import AssocRulesModel
from src.evaluation.survey_generator import generate_assoc_rules_survey

print("="*60)
print("  BUOC 8: DANH GIA ASSOCIATION RULES")
print("="*60)

# --- Bước 1: Load data ---
data_path = os.path.join(PROCESSED_DIR, "order_products.parquet")
products_path = os.path.join(PROCESSED_DIR, "products.parquet")
if not os.path.exists(data_path):
    print("ERROR: Chua co data! Chay scripts/01_load_data.py truoc.")
    sys.exit(1)

print("\n1. Loading data...")
order_products = pd.read_parquet(data_path)
products = pd.read_parquet(products_path)
print(f"   -> {len(order_products)} records, {len(products)} products")

# --- Bước 2: Load AssocRules ---
assoc_rules_path = os.path.join(MODEL_DIR, "assoc_rules")
rules_file = os.path.join(assoc_rules_path, "rules.csv")
if not os.path.exists(rules_file):
    print("ERROR: Chua co rules! Chay scripts/06_assoc_rules.py truoc.")
    sys.exit(1)

print("\n2. Loading AssocRulesModel...")
arm = AssocRulesModel()
arm.load(assoc_rules_path)
print(f"   Tong so rules: {len(arm.rules_df)}")

# --- Bước 3: Thống kê rules ---
print("\n3. Thong ke rules:")
print(f"   Tong so rules: {len(arm.rules_df)}")
if not arm.rules_df.empty:
    print(f"   Support  - min: {arm.rules_df['support'].min():.6f}, "
          f"max: {arm.rules_df['support'].max():.6f}, "
          f"mean: {arm.rules_df['support'].mean():.6f}")
    print(f"   Confidence - min: {arm.rules_df['confidence'].min():.4f}, "
          f"max: {arm.rules_df['confidence'].max():.4f}, "
          f"mean: {arm.rules_df['confidence'].mean():.4f}")
    print(f"   Lift - min: {arm.rules_df['lift'].min():.4f}, "
          f"max: {arm.rules_df['lift'].max():.4f}, "
          f"mean: {arm.rules_df['lift'].mean():.4f}")
    
    # Phân bố lift
    lift_bins = [1, 2, 5, 10, 50, 100, 1000]
    print("   Phan bo lift:")
    for i in range(len(lift_bins) - 1):
        lo, hi = lift_bins[i], lift_bins[i+1]
        count = ((arm.rules_df['lift'] >= lo) & (arm.rules_df['lift'] < hi)).sum()
        if count > 0:
            print(f"     {lo:>4} - {hi:>4}: {count}")
    count_ge = (arm.rules_df['lift'] >= lift_bins[-1]).sum()
    if count_ge > 0:
        print(f"     >= {lift_bins[-1]}: {count_ge}")

# --- Bước 4: Test recommend cho sample products ---
print("\n4. Test recommend cho sample products:")
name_map = dict(zip(products['product_id'], products['product_name']))
sample_ids = products['product_id'].sample(n=10, random_state=RANDOM_SEED).tolist()

for pid_a in sample_ids:
    pname_a = name_map.get(pid_a, "?")
    recs = arm.recommend(pid_a, top_k=10)
    print(f"\n   [{pid_a}] {pname_a}:")
    if recs:
        for pid_b, lift in recs[:5]:  # Chỉ in top-5
            pname_b = name_map.get(pid_b, "?")
            print(f"     -> [{pid_b}] {pname_b} (lift={lift:.4f})")
    else:
        print("     (khong co recommend)")

# --- Bước 5: Sinh survey samples ---
print("\n5. Sinh survey samples...")
survey_path = generate_assoc_rules_survey(
    model=arm,
    products_df=products,
    model_name="assoc_rules",
    n_samples=100,
    top_k=100,
    seed=RANDOM_SEED,
)

# --- Bước 6: Tổng hợp kết quả ---
print("\n6. Luu ket qua evaluation...")
eval_dir = os.path.join(RESULT_DIR, "assoc_rules_evaluation")
os.makedirs(eval_dir, exist_ok=True)

# Thống kê rules
stats = {
    'n_rules': len(arm.rules_df) if arm.rules_df is not None else 0,
    'n_products_with_rules': (
        arm.rules_df['antecedent'].nunique()
        if arm.rules_df is not None and not arm.rules_df.empty
        else 0
    ),
    'support_min': float(arm.rules_df['support'].min()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'support_max': float(arm.rules_df['support'].max()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'support_mean': float(arm.rules_df['support'].mean()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'confidence_min': float(arm.rules_df['confidence'].min()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'confidence_max': float(arm.rules_df['confidence'].max()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'confidence_mean': float(arm.rules_df['confidence'].mean()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'lift_min': float(arm.rules_df['lift'].min()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'lift_max': float(arm.rules_df['lift'].max()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'lift_mean': float(arm.rules_df['lift'].mean()) if arm.rules_df is not None and not arm.rules_df.empty else 0,
    'n_survey_samples': len(pd.read_csv(survey_path)) if os.path.exists(survey_path) else 0,
    'survey_file': survey_path,
}

with open(os.path.join(eval_dir, "evaluation_stats.json"), 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n   Da luu ket qua tai: {eval_dir}")
print("\n Done!")