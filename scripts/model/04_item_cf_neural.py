"""
Bước 4: Item2Vec (Neural Item-Based CF) — Word2Vec Skip-gram trên giỏ hàng.
Chạy riêng: python scripts/model/04_item_cf_neural.py
Yêu cầu: scripts/model/01_load_data.py đã chạy
Output: models/item2vec/ (word2vec.model + mapping.json)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.item_cf_neural import ItemCFNeuralModel

print("="*60)
print("  BUOC 4: ITEM2VEC (NEURAL CF)")
print("="*60)

# Load data da cache
data_path = os.path.join(PROCESSED_DIR, "order_products.parquet")
products_path = os.path.join(PROCESSED_DIR, "products.parquet")
if not os.path.exists(data_path):
    print("ERROR: Chua co data! Chay scripts/model/01_load_data.py truoc.")
    sys.exit(1)

print("\n1. Loading data...")
order_products = pd.read_parquet(data_path)
products = pd.read_parquet(products_path)
print(f"   -> {len(order_products)} records, {len(products)} products")

# Kiem tra neu da train
save_path = os.path.join(MODEL_DIR, "item2vec")
if os.path.exists(os.path.join(save_path, "word2vec.model")):
    print("\n2. Item2Vec da train, loading...")
    i2v = ItemCFNeuralModel()
    i2v.load(save_path)
else:
    print("\n2. Training Item2Vec...")
    i2v = ItemCFNeuralModel()
    i2v.fit(order_products, products)
    i2v.save(save_path)

# Test thu
sample_id = products['product_id'].iloc[0]
pname = products[products['product_id'] == sample_id]['product_name'].values[0]
print(f"\n3. Test recommend cho [{sample_id}] {pname}:")
recs = i2v.recommend(sample_id, top_k=5)
for pid, sim in recs:
    rname = products[products['product_id'] == pid]['product_name'].values
    rname = rname[0] if len(rname) else "?"
    print(f"   -> {pid}: {rname} (sim={sim:.4f})")

print("\n Done!")