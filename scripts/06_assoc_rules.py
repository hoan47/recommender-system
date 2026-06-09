"""
Bước 6: Association Rules — từ co-occurrence matrix.
Chạy riêng: python scripts/06_assoc_rules.py
Yêu cầu: scripts/01_load_data.py + scripts/03_ochiai.py đã chạy
Output: models/assoc_rules/ (rules.csv + metadata.json)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.ochiai import OchiaiModel
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

# Kiem tra Ochiai da train chua
ochiai_path = os.path.join(MODEL_DIR, "ochiai")
if not os.path.exists(os.path.join(ochiai_path, "cooc_matrix.npz")):
    print("ERROR: Chua co OchiaiModel! Chay scripts/03_ochiai.py truoc.")
    sys.exit(1)

print("\n1. Loading data...")
order_products = pd.read_parquet(data_path)
products = pd.read_parquet(products_path)
print(f"   -> {len(order_products)} records, {len(products)} products")

print("\n2. Loading OchiaiModel...")
ochiai = OchiaiModel()
ochiai.load(ochiai_path)

# Kiem tra neu da train
save_path = os.path.join(MODEL_DIR, "assoc_rules")
if os.path.exists(os.path.join(save_path, "rules.csv")):
    print("\n3. AssocRules da train, loading...")
    arm = AssocRulesModel()
    arm.load(save_path)
else:
    print("\n3. Training AssocRules...")
    arm = AssocRulesModel()
    arm.fit(ochiai, order_products)
    arm.save(save_path)

print("\n Done!")