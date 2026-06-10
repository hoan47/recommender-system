"""
DeepWalk: Graph-based embedding.
Xây đồ thị sản phẩm dựa trên co-occurrence, học embedding qua uniform random walk.
Chạy riêng: python scripts/05_deepwalk.py
Yêu cầu: scripts/01_load_data.py đã chạy
Output: models/deepwalk/ (embeddings.npy + metadata.json + word2vec.model)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.deepwalk import DeepWalkModel

print("="*60)
print("  BUOC 5: DEEPWALK (GRAPH EMBEDDING)")
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
save_path = os.path.join(MODEL_DIR, "deepwalk")
if os.path.exists(os.path.join(save_path, "embeddings.npy")):
    print("\n2. DeepWalk da train, loading...")
    model = DeepWalkModel()
    model.load(save_path)
else:
    print("\n2. Training DeepWalk...")
    model = DeepWalkModel()
    model.fit(order_products, products)
    model.save(save_path)
    print(f"   -> Saved to {save_path}")

# Test thu
sample_id = products['product_id'].iloc[0]
pname = products[products['product_id'] == sample_id]['product_name'].values[0]
print(f"\n3. Test recommend cho [{sample_id}] {pname}:")
recs = model.recommend(sample_id, top_k=5)
for pid, score in recs:
    rname = products[products['product_id'] == pid]['product_name'].values
    rname = rname[0] if len(rname) else "?"
    print(f"   -> {pid}: {rname} (score={score:.4f})")

print("\n Done!")