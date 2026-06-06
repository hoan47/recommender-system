"""
Trình tải dữ liệu tập trung cho dataset Instacart Market Basket Analysis
Cung cấp hàm load cho: products, prior interactions, train/test split
"""

import gc
import pandas as pd
from src.config import DATA_DIR

# Ánh xạ tên cột → kiểu dữ liệu nhỏ nhất (tiết kiệm RAM)
CASTS = {
    'order_id': 'int32',
    'product_id': 'int32',
    'user_id': 'int32',
    'add_to_cart_order': 'int8',
    'reordered': 'int8',
}

def _cast(df):
    """Ép các cột số sang dtype nhỏ nhất để giảm dung lượng RAM"""
    for c, t in CASTS.items():
        if c in df.columns:
            df[c] = df[c].astype(t)
    return df

def load_products():
    """
    Tải products.csv và join với departments.csv
    Trả về DataFrame gồm: product_id, product_name, department_id, department
    """
    # Tải products và departments riêng
    p = pd.read_csv(DATA_DIR / "products.csv", encoding="utf-8")
    d = pd.read_csv(DATA_DIR / "departments.csv", encoding="utf-8")
    _cast(p); _cast(d)
    # Merge để lấy tên department (aisles.csv không được dùng)
    m = p.merge(d, on="department_id", how="left")
    # Xử lý missing values tập trung
    m["product_name"] = m["product_name"].fillna("unknown")
    m["department"] = m["department"].fillna("unknown")
    gc.collect()
    return m[["product_id", "product_name", "department_id", "department"]]

def load_prior():
    """
    Tải order_products__prior.csv
    Trả về DataFrame: order_id, product_id, add_to_cart_order, reordered
    32.4M records — đã ép kiểu để giảm RAM
    """
    df = pd.read_csv(DATA_DIR / "order_products__prior.csv", encoding="utf-8")
    _cast(df)
    return df

def load_train_test():
    """
    Tách train/test từ order_products__train.csv dựa trên orders.csv[eval_set]
    Dataset KHÔNG có file order_products__test.csv riêng
    Trả về (train_df, test_df) với các cột: order_id, product_id
    """
    # Tải orders để biết order_id nào thuộc train/test
    orders = pd.read_csv(DATA_DIR / "orders.csv", encoding="utf-8")
    _cast(orders)
    train_ids = orders[orders["eval_set"] == "train"]["order_id"]
    test_ids = orders[orders["eval_set"] == "test"]["order_id"]
    del orders; gc.collect()
    # Tải tất cả interactions từ file train
    t = pd.read_csv(DATA_DIR / "order_products__train.csv", encoding="utf-8")
    _cast(t)
    # Tách dựa trên order_id
    train = t[t["order_id"].isin(train_ids)].copy()
    test = t[t["order_id"].isin(test_ids)].copy()
    del t; gc.collect()
    return train, test