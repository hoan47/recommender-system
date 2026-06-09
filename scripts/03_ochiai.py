"""
Bước 3: Ochiai + Confidence Score.
Chạy riêng: python scripts/03_ochiai.py
Yêu cầu: scripts/01_load_data.py đã chạy
Output: models/ochiai/ (cooc_matrix.npz + metadata.json)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.ochiai import OchiaiModel

print("="*60)
print("  BUOC 3: OCHIAI + CONFIDENCE SCORE")
print("="*60)

# Load data da cache
data_path = os.path.join(PROCESSED_DIR, "order_products.parquet")
products_path = os.path.join(PROCESSED_DIR, "products.parquet")
if not os.path.exists(data_path):
    print("ERROR: Chua co data! Chay scripts/01_load_data.py truoc.")
    sys.exit(1)

print("\n1. Loading data...")
order_products = pd.read_parquet(data_path)
products = pd.read_parquet(products_path)
print(f"   -> {len(order_products)} records, {len(products)} products")

# Kiem tra neu da train
save_path = os.path.join(MODEL_DIR, "ochiai")
if os.path.exists(os.path.join(save_path, "cooc_matrix.npz")):
    print("\n2. OchiaiModel da train, loading...")
    ochiai = OchiaiModel()
    ochiai.load(save_path)
else:
    print("\n2. Training OchiaiModel (co the mat vai phut)...")
    ochiai = OchiaiModel()
    ochiai.fit(order_products, products)
    ochiai.save(save_path)
    print(f"   -> Saved to {save_path}")

# Test thu
sample_id = products['product_id'].iloc[0]
pname = products[products['product_id'] == sample_id]['product_name'].values[0]
print(f"\n3. Test recommend cho [{sample_id}] {pname}:")
recs = ochiai.recommend(sample_id, top_k=5)
for pid, score in recs:
    rname = products[products['product_id'] == pid]['product_name'].values
    rname = rname[0] if len(rname) else "?"
    print(f"   -> {pid}: {rname} (score={score:.4f})")

print("\n Done!")