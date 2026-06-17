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
from src.features.product_filter import (
    get_excluded_product_ids,
    filter_order_products,
    get_filter_stats,
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("="*60)
print("  BƯỚC 1: LOAD DATA + FILTER STRATEGY")
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

# --- Filter Strategy: loại non-food products ---
print("\n3. Filtering non-food products...")
excluded_ids = get_excluded_product_ids(products)

# Thống kê trước khi lọc
get_filter_stats(products, order_products, excluded_ids)

# Lọc order_products
order_products, removed = filter_order_products(order_products, excluded_ids)
print(f"   -> Đã loại {removed:,} records ({removed/len(order_products)*100:.1f}% nếu tính trên dữ liệu sau lọc)")
print(f"   -> Còn lại: {len(order_products):,} records")
print(f"   -> Số đơn hàng còn lại: {order_products['order_id'].nunique():,}")

# Lưu dạng parquet (nén tốt hơn CSV)
order_products.to_parquet(os.path.join(PROCESSED_DIR, "order_products.parquet"))
print(f"   -> Saved to {PROCESSED_DIR}order_products.parquet")

print("\n Done! 2 files saved (order_products.parquet đã được lọc).")
