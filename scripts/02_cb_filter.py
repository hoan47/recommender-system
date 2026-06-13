"""
Bước 2: CB Filter — Content-Based Diversity Filter.
Chạy riêng: python scripts/02_cb_filter.py
Yêu cầu: scripts/01_load_data.py đã chạy (có products.parquet)
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
print("  BƯỚC 2: CB FILTER")
print("="*60)

# Load data đã cache
products_path = os.path.join(PROCESSED_DIR, "products.parquet")
if not os.path.exists(products_path):
    print("ERROR: Chưa có products.parquet! Chạy scripts/01_load_data.py trước.")
    sys.exit(1)

print("\n1. Loading products...")
products = pd.read_parquet(products_path)
print(f"   -> {len(products)} products (EN)")

# Gộp thêm tên tiếng Việt nếu có
products_vi_path = os.path.join(PROCESSED_DIR, "products_vi.csv")
if os.path.exists(products_vi_path):
    print("   Gộp thêm product_name_vi từ products_vi.csv...")
    products_vi = pd.read_csv(products_vi_path, encoding='utf-8')
    products = products.merge(products_vi, on='product_id', how='left')
    print(f"   -> Đã gộp {products_vi.shape[0]} bản ghi tiếng Việt")
else:
    print("   (Không tìm thấy products_vi.csv — chỉ dùng product_name EN)")

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