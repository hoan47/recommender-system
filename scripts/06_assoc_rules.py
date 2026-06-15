"""
Bước 6: Association Rules — từ co-occurrence matrix.
Chạy riêng: python scripts/06_assoc_rules.py
Yêu cầu: scripts/01_load_data.py + scripts/03_item_cf.py đã chạy
Output: models/assoc_rules/ (rules.csv + metadata.json)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.item_cf import ItemCFModel
from src.models.assoc_rules import AssocRulesModel

print("="*60)
print("  BUOC 6: ASSOCIATION RULES")
print("="*60)

# Load data da cache
data_path = os.path.join(PROCESSED_DIR, "order_products.parquet")
products_path = os.path.join(PROCESSED_DIR, "products.parquet")
if not os.path.exists(data_path):
    print("ERROR: Chua co data! Chay scripts/01_load_data.py truoc.")
    sys.exit(1)

# Kiem tra Item-CF da train chua
item_cf_path = os.path.join(MODEL_DIR, "item_cf")
if not os.path.exists(os.path.join(item_cf_path, "cooc_matrix.npz")):
    print("ERROR: Chua co ItemCFModel! Chay scripts/03_item_cf.py truoc.")
    sys.exit(1)

print("\n1. Loading data...")
order_products = pd.read_parquet(data_path)
products = pd.read_parquet(products_path)
print(f"   -> {len(order_products)} records, {len(products)} products")

print("\n2. Loading ItemCFModel...")
item_cf = ItemCFModel()
item_cf.load(item_cf_path)

# Kiem tra neu da train
save_path = os.path.join(MODEL_DIR, "assoc_rules")
if os.path.exists(os.path.join(save_path, "rules.csv")):
    print("\n3. AssocRules da train, loading...")
    arm = AssocRulesModel()
    arm.load(save_path)
else:
    print("\n3. Training AssocRules...")
    arm = AssocRulesModel()
    arm.fit(item_cf, order_products)
    arm.save(save_path)

# Test thu
sample_id = products['product_id'].iloc[0]
pname = products[products['product_id'] == sample_id]['product_name'].values[0]
print(f"\n4. Test recommend cho [{sample_id}] {pname}:")
recs = arm.recommend(sample_id, top_k=5)
for pid, lift in recs:
    rname = products[products['product_id'] == pid]['product_name'].values
    rname = rname[0] if len(rname) else "?"
    print(f"   -> {pid}: {rname} (lift={lift:.4f})")

print("\n Done!")
