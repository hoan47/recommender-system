"""
Lấy tất cả unique product IDs từ cả cột product_A_id và product_B_id
trong file gemini_responses_filtered.csv, lưu vào results/unique_product_ids.txt
"""
import csv

INPUT_FILE = 'data/survey/llm_raw_responses/gemini_responses_filtered.csv'
OUTPUT_FILE = 'results/unique_product_ids.txt'

unique_ids = set()

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        unique_ids.add(row['product_A_id'])
        unique_ids.add(row['product_B_id'])

# Sắp xếp theo số
sorted_ids = sorted(unique_ids, key=int)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(sorted_ids))

print(f'Tổng unique product IDs: {len(sorted_ids)}')
print(f'Đã lưu vào {OUTPUT_FILE}')