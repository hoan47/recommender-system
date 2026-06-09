"""
Bước 7: Ensemble — Co-occurrence Ensemble + CB Filter.
Chạy riêng: python scripts/07_ensemble.py
Yêu cầu: các scripts 01-06 đã chạy
Output: in kết quả recommend cho 4 sản phẩm mẫu
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import scipy.sparse

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.cb_filter import CBFilter
from src.models.ochiai import OchiaiModel
from src.models.item2vec import Item2VecModel
from src.models.node2vec import Node2VecModel
from src.models.ensemble import EnsembleModel

print("="*60)
print("  BUOC 7: ENSEMBLE + CB FILTER")
print("="*60)

# Kiem tra cac model da train chua
checks = [
    ("CB Filter", os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")),
    ("Ochiai",    os.path.join(MODEL_DIR, "ochiai", "cooc_matrix.npz")),
    ("Item2Vec",  os.path.join(MODEL_DIR, "item2vec", "word2vec.model")),
    ("Node2Vec",  os.path.join(MODEL_DIR, "node2vec", "embeddings.npy")),
]
for name, path in checks:
    if not os.path.exists(path):
        print(f"ERROR: Thieu {name}! Chay scripts tuong ung truoc.")
        sys.exit(1)

print("\n1. Loading data...")
products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))

print("\n2. Loading models...")

print("   CB Filter...")
cb = CBFilter()
# Load product vectors tu file
product_vectors = scipy.sparse.load_npz(
    os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")
)
cb.product_vectors = product_vectors
cb.product_id_to_idx = {int(pid): i for i, pid in enumerate(products['product_id'])}
print(f"   -> {len(cb.product_id_to_idx)} products")

print("   Ochiai...")
ochiai = OchiaiModel()
ochiai.load(os.path.join(MODEL_DIR, "ochiai"))

print("   Item2Vec...")
i2v = Item2VecModel()
i2v.load(os.path.join(MODEL_DIR, "item2vec"))

print("   Node2Vec...")
n2v = Node2VecModel()
n2v.load(os.path.join(MODEL_DIR, "node2vec"))

print("\n3. Initializing Ensemble...")
ensemble = EnsembleModel()
ensemble.fit(ochiai, i2v, n2v, cb)

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