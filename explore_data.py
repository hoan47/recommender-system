import csv, os
from collections import Counter

files = ['aisles.csv','departments.csv','products.csv','orders.csv','order_products__train.csv','order_products__test.csv']

for fname in files:
    path = os.path.join('data', fname)
    print('='*60)
    print(f'FILE: {fname}')
    print('='*60)
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'Rows: {len(rows)}')
    print(f'Cols: {len(reader.fieldnames)}')
    print(f'Fields: {reader.fieldnames}')
    
    for col in reader.fieldnames:
        vals = [r[col] for r in rows]
        non_null = [v for v in vals if v != '']
        nulls = len(vals) - len(non_null)
        uniq = len(set(non_null))
        print(f'  [{col}] empty={nulls} unique={uniq}', end='')
        if uniq <= 10:
            c = Counter(non_null)
            print(f'  values={dict(c.most_common())}')
        else:
            try:
                nums = [float(v) for v in non_null]
                print(f'  range={min(nums):.2f}-{max(nums):.2f}  avg={sum(nums)/len(nums):.2f}')
            except:
                sample = list(set(non_null))[:3]
                print(f'  sample={sample}')
    print()