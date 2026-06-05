"""
loadata.py — Load & prepare dữ liệu cho 4 models
  - Content-Based (CB)
  - Collaborative Filtering (SPMI)
  - Knowledge Graph (RWR)
  - Hybrid
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_data(data_dir: str = DATA_DIR) -> dict:
    """Load 6 CSV files thành DataFrame."""

    files = {
        'aisles':           'aisles.csv',
        'departments':      'departments.csv',
        'products':         'products.csv',
        'orders':           'orders.csv',
        'order_products__train': 'order_products__train.csv',
        'order_products__test':  'order_products__test.csv',
    }

    data = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        data[key] = pd.read_csv(path)
        print(f"  OK {fname:30s} -> {len(data[key]):>8,} rows")

    return data


def clean_data(data: dict) -> dict:
    """Làm sạch & gộp dữ liệu, trả về dict sẵn sàng cho 4 model."""

    # 1. Products gắn aisle_name, department_name
    products = data['products'].merge(data['aisles'], on='aisle_id', how='left') \
                               .merge(data['departments'], on='department_id', how='left')

    # 2. Orders: xử lý days_since_prior_order NaN → 0
    orders = data['orders'].copy()
    orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)

    # 3. Gộp train + test
    op_train = data['order_products__train'].copy()
    op_train['set'] = 'train'
    op_test = data['order_products__test'].copy()
    op_test['set'] = 'test'
    order_products = pd.concat([op_train, op_test], ignore_index=True)

    # Gắn thông tin order (user_id, order_dow, ...)
    order_products = order_products.merge(
        orders[['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day']],
        on='order_id', how='left'
    )

    return {
        'products':        products,
        'orders':          orders,
        'order_products':  order_products,
        'train_only':      op_train,
        'test_only':       op_test,
        'users':           orders[['user_id']].drop_duplicates().sort_values('user_id').reset_index(drop=True),
    }


def prepare(data_dir: str = DATA_DIR) -> dict:
    """Pipeline chính: load → clean → trả về dict cho 4 model."""

    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    raw = load_data(data_dir)

    print("\n" + "=" * 50)
    print("CLEANING DATA")
    print("=" * 50)
    result = clean_data(raw)

    print(f"\nDone! Data ready for 4 models:")
    print(f"   Products      : {len(result['products']):>8,}")
    print(f"   Orders        : {len(result['orders']):>8,}")
    print(f"   OrderProducts : {len(result['order_products']):>8,}")
    print(f"   Users         : {len(result['users']):>8,}")
    print(f"   Train set     : {len(result['train_only']):>8,}")
    print(f"   Test set      : {len(result['test_only']):>8,}")
    print()

    return result


if __name__ == '__main__':
    data = prepare()