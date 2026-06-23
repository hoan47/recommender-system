"""
Top sản phẩm có nhiều gợi ý nhất từ file gemini_responses_filtered.csv
Đếm tần suất product_A_id, join với products.parquet lấy tên, in ra console.

Usage:
    python scripts/15_top_suggested_products.py [n]
    n: số lượng top cần lấy (mặc định 100)
"""
import csv
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DIR

# --- params ---
n = 100
if len(sys.argv) > 1:
    n = int(sys.argv[1])

INPUT_CSV = 'data/survey/llm_raw_responses/gemini_responses_filtered.csv'
PRODUCTS_PARQUET = os.path.join(PROCESSED_DIR, 'products.parquet')

# --- đếm tần suất product_A_id ---
counts = {}
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row['product_A_id']
        counts[pid] = counts.get(pid, 0) + 1

# --- load products.parquet lấy product_id + product_name ---
products = pd.read_parquet(PRODUCTS_PARQUET, columns=['product_id', 'product_name'])
pid_to_name = dict(zip(products['product_id'].astype(str), products['product_name']))

# --- sort, lấy top n ---
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

# --- in ra console ---
print(f'Top {n} sản phẩm có nhiều gợi ý nhất (product_A_id):')
print(f'{"#":>4} | {"ID":>8} | {"Tên sản phẩm":<60} | Số gợi ý')
print('-' * 90)
for rank, (pid, cnt) in enumerate(sorted_counts, 1):
    name = pid_to_name.get(pid, '?')
    print(f'{pid:>8} | {name:<60} | {cnt}')