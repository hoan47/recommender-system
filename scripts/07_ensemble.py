"""
Bước 7: Ensemble — Co-occurrence Ensemble + CB Filter.
Chạy riêng: python scripts/07_ensemble.py
Yêu cầu: các scripts 01-06 đã chạy
Output: in kết quả recommend cho 4 sản phẩm mẫu
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import scipy.sparse

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.cb_filter import CBFilter
from src.models.item_cf import ItemCFModel
from src.models.item2vec import Item2VecModel
from src.models.metapath2vec import Metapath2VecModel
from src.models.ensemble import EnsembleModel

print("="*60)
print("  BUOC 7: ENSEMBLE + CB FILTER")
print("="*60)

# Kiem tra cac model da train chua
checks = [
    ("CB Filter", os.path.join(MODEL_DIR, "cb_filter", "tfidf_vectors.npz")),
    ("Item-CF",   os.path.join(MODEL_DIR, "item_cf", "cooc_matrix.npz")),
    ("Item2Vec",  os.path.join(MODEL_DIR, "item2vec", "word2vec.model")),
    ("Metapath2Vec",  os.path.join(MODEL_DIR, "metapath2vec", "embeddings.npy")),
]
for name, path in checks:
    if not os.path.exists(path):
        print(f"ERROR: Thieu {name}! Chay scripts tuong ung truoc.")
        sys.exit(1)

print("\n1. Loading data...")
products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))

print("\n2. Loading models...")

print("   CB Filter (ensemble Count + TF-IDF)...")
cb = CBFilter()
# Load TF-IDF vectors
product_vectors_tfidf = scipy.sparse.load_npz(
    os.path.join(MODEL_DIR, "cb_filter", "tfidf_vectors.npz")
)
cb.product_vectors = product_vectors_tfidf  # setter gán vào product_vectors_tfidf
# Load Count vectors (L2-normalized)
count_path = os.path.join(MODEL_DIR, "cb_filter", "count_vectors.npz")
if os.path.exists(count_path):
    cb.product_vectors_count = scipy.sparse.load_npz(count_path)
    print(f"   -> Loaded Count vectors: {cb.product_vectors_count.shape}")
# Load product_id_to_idx đúng mapping từ file (tránh lệch index so với lúc fit)
with open(os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")) as f:
    cb.product_id_to_idx = {int(k): v for k, v in json.load(f).items()}
print(f"   -> {len(cb.product_id_to_idx)} products")

print("   Item-CF...")
item_cf = ItemCFModel()
item_cf.load(os.path.join(MODEL_DIR, "item_cf"))

print("   Item2Vec...")
i2v = Item2VecModel()
i2v.load(os.path.join(MODEL_DIR, "item2vec"))

print("   Metapath2Vec...")
metapath2vec = Metapath2VecModel()
metapath2vec.load(os.path.join(MODEL_DIR, "metapath2vec"))

print("\n3. Initializing Ensemble...")
ensemble = EnsembleModel()
ensemble.fit(item_cf, i2v, metapath2vec, cb)

# Save ensemble model de 08_streamlit_app.py load 1 lan
print("\n   Saving Ensemble model...")
ensemble.save()

print("\n4. Testing recommendations cho 4 san pham mau:")
test_products = [1, 250, 5000, 25000]

for pid in test_products:
    pname = products[products['product_id'] == pid]['product_name'].values
    pname = pname[0] if len(pname) else f"Product {pid}"

    print(f"\n   --- [{pid}] {pname} ---")

    # Without CB Filter
    recs_no_cb = ensemble.recommend(pid, use_cb_filter=False)
    print(f"   Without CB Filter (top-5):")
    for rid, score in recs_no_cb[:5]:
        rname = products[products['product_id'] == rid]['product_name'].values
        rname = rname[0] if len(rname) else "?"
        print(f"     -> {rid}: {rname} (score={score:.4f})")

    # With CB Filter
    recs_cb = ensemble.recommend(pid, use_cb_filter=True)
    print(f"   With CB Filter (top-5):")
    for rid, score in recs_cb[:5]:
        rname = products[products['product_id'] == rid]['product_name'].values
        rname = rname[0] if len(rname) else "?"
        print(f"     -> {rid}: {rname} (score={score:.4f})")

print("\n Done!")