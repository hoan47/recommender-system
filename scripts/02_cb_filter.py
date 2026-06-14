"""
Bước 2: CB Filter — Content-Based Diversity Filter (tiếng Việt).
Chạy riêng: python scripts/02_cb_filter.py
Yêu cầu: data/processed/products_vi.csv (phải có sẵn)
Output: models/cb_filter/ (product_vectors.npz)
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import scipy.sparse

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.cb_filter import CBFilter

print("="*60)
print("  BƯỚC 2: CB FILTER (tiếng Việt)")
print("="*60)

# Load data tiếng Việt
products_path = os.path.join(PROCESSED_DIR, "products_vi.csv")
if not os.path.exists(products_path):
    print(f"ERROR: Chưa có {products_path}!")
    print("Yêu cầu: phải có file products_vi.csv với cột product_name_vi")
    sys.exit(1)

print("\n1. Loading products (tiếng Việt)...")
products = pd.read_csv(products_path)
print(f"   -> {len(products)} products (VI)")
print(f"   -> Cột product_name_vi có sẵn, dùng làm TF-IDF")

print("\n2. Fitting CBFilter...")
cb = CBFilter()
cb.fit(products)

# Save
save_path = os.path.join(MODEL_DIR, "cb_filter")
os.makedirs(save_path, exist_ok=True)
scipy.sparse.save_npz(os.path.join(save_path, "product_vectors.npz"), cb.product_vectors)

# Lưu product_id_to_idx để 07_ensemble.py load đúng mapping
with open(os.path.join(save_path, "product_id_to_idx.json"), 'w') as f:
    json.dump({str(k): v for k, v in cb.product_id_to_idx.items()}, f)

print(f"\n   -> Saved to {save_path}")
print(f"   -> {len(cb.product_id_to_idx)} products vectorized")
print(f"   -> Vector shape: {cb.product_vectors.shape}")
print("\n Done!")