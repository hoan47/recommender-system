"""
Bước 1: Load và cache dữ liệu — toàn bộ 49,688 products (KHÔNG lọc non-food).
Chạy riêng: python scripts/model/01_load_data.py
Output: data/processed/ (các file đã xử lý)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import PROCESSED_DIR
from src.features.loader import load_products, load_order_products

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("="*60)
print("  BƯỚC 1: LOAD DATA — 49,688 products, 33.8M records")
print("="*60)

print("\n1. Loading products...")
products = load_products()
print(f"   -> {len(products)} products")

print("\n2. Loading order_products (prior + train)...")
order_products = load_order_products(use_prior=True, use_train=True)
print(f"   -> {len(order_products):,} records")
print(f"   -> {order_products['order_id'].nunique():,} unique orders")

# Lưu dạng parquet — KHÔNG lọc, giữ toàn bộ 49,688 products
products.to_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
print(f"\n3. Saved {PROCESSED_DIR}products.parquet ({len(products)} products)")

order_products.to_parquet(os.path.join(PROCESSED_DIR, "order_products.parquet"))
print(f"   Saved {PROCESSED_DIR}order_products.parquet ({len(order_products):,} records)")

print(f"\n Done! 2 files saved (toàn bộ dữ liệu gốc, không lọc non-food).")