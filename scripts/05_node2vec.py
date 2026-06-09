"""
Bước 5: Node2Vec — Graph embedding với random walk.
Chạy riêng: python scripts/05_node2vec.py
Yêu cầu: scripts/01_load_data.py đã chạy
Output: models/node2vec/ (embeddings.npy + metadata.json)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.node2vec import Node2VecModel

print("="*60)
print("  BUOC 5: NODE2VEC (GRAPH EMBEDDING)")
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
save_path = os.path.join(MODEL_DIR, "node2vec")
if os.path.exists(os.path.join(save_path, "embeddings.npy")):
    print("\n2. Node2Vec da train, loading...")
    n2v = Node2VecModel()
    n2v.load(save_path)
else:
    print("\n2. Training Node2Vec (co the mat nhieu phut)...")
    n2v = Node2VecModel()
    n2v.fit(order_products, products)
    n2v.save(save_path)

# Test thu
sample_id = products['product_id'].iloc[0]
pname = products[products['product_id'] == sample_id]['product_name'].values[0]
print(f"\n3. Test recommend cho [{sample_id}] {pname}:")
recs = n2v.recommend(sample_id, top_k=5)
for pid, sim in recs:
    rname = products[products['product_id'] == pid]['product_name'].values
    rname = rname[0] if len(rname) else "?"
    print(f"   -> {pid}: {rname} (sim={sim:.4f})")

print("\n Done!")