"""
Bước 1: Load và cache dữ liệu.
Chạy riêng: python scripts/01_load_data.py
Output: data/processed/ (các file đã xử lý)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_DIR
from src.features.loader import load_products, load_order_products

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("="*60)
print("  BƯỚC 1: LOAD DATA")
print("="*60)

print("\n1. Loading products...")
products = load_products()
print(f"   -> {len(products)} products")
products.to_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
print(f"   -> Saved to {PROCESSED_DIR}products.parquet")

print("\n2. Loading order_products (prior + train)...")
order_products = load_order_products(use_prior=True, use_train=True)
print(f"   -> {len(order_products)} records")
print(f"   -> {order_products['order_id'].nunique()} unique orders")

# Lưu dạng parquet (nén tốt hơn CSV)
order_products.to_parquet(os.path.join(PROCESSED_DIR, "order_products.parquet"))
print(f"   -> Saved to {PROCESSED_DIR}order_products.parquet")

print("\n Done! 2 files saved.")