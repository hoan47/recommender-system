"""
Trình tải dữ liệu tập trung cho dataset Instacart Market Basket Analysis
Cung cấp hàm load cho: products, prior interactions, train/test split
"""

import gc
import numpy as np
import pandas as pd
from collections import defaultdict
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


# ===== Temporal user-based split cho evaluation =====
# 80% order đầu của mỗi user → train, 20% order cuối → test
# Đảm bảo không có data leakage

def _build_user_orders(orders_df):
    """
    Xây dựng dict: user_id → list of (order_number, order_id) sorted by order_number
    """
    user_orders = defaultdict(list)
    for _, row in orders_df.iterrows():
        user_orders[row["user_id"]].append((row["order_number"], row["order_id"]))
    # Sắp xếp theo order_number tăng dần
    for uid in user_orders:
        user_orders[uid].sort(key=lambda x: x[0])
    return user_orders


def _split_user_orders(user_orders, train_ratio=0.8):
    """
    Với mỗi user: 80% order đầu → train, 20% cuối → test
    Trả về (train_order_ids, test_order_ids)
    """
    train_ids = []
    test_ids = []
    for uid, orders in user_orders.items():
        n = len(orders)
        split_idx = int(n * train_ratio)
        if split_idx >= n:
            split_idx = n - 1  # Giữ ít nhất 1 order cho test
        if split_idx < 1:
            split_idx = 1  # Giữ ít nhất 1 order cho train
        for i, (_, oid) in enumerate(orders):
            if i < split_idx:
                train_ids.append(oid)
            else:
                test_ids.append(oid)
    return set(train_ids), set(test_ids)


def load_temporal_test_cases(train_ratio=0.8):
    """
    Tạo test cases từ temporal user-based split.
    - 80% order đầu của mỗi user → train (xây model)
    - 20% order cuối → test (đánh giá)
    
    Trả về:
        test_cases: list of (seed_product, ground_truth_products)
        n_prior_orders: số order dùng để xây model (prior + train split)
    """
    print("  [DataLoader] Loading orders.csv ...")
    orders = pd.read_csv(DATA_DIR / "orders.csv", encoding="utf-8")
    _cast(orders)
    
    # Xây user_orders từ tất cả orders (prior + train + test)
    user_orders = _build_user_orders(orders)
    
    # Split temporal
    train_order_ids, test_order_ids = _split_user_orders(user_orders, train_ratio)
    
    # Prior orders (eval_set='prior') dùng để xây model
    prior_ids = set(orders[orders["eval_set"] == "prior"]["order_id"].values)
    del orders; gc.collect()
    
    # Tải prior interactions
    print("  [DataLoader] Loading order_products__prior.csv ...")
    prior_df = load_prior()
    
    # Tải train interactions (chứa ground truth cho cả train và test split)
    print("  [DataLoader] Loading order_products__train.csv ...")
    train_all = pd.read_csv(DATA_DIR / "order_products__train.csv", encoding="utf-8")
    _cast(train_all)
    
    # Gộp prior + train split để xây model
    train_orders = prior_ids | train_order_ids
    model_df = pd.concat([
        prior_df[prior_df["order_id"].isin(train_orders)],
        train_all[train_all["order_id"].isin(train_orders)]
    ], ignore_index=True)
    del prior_df; gc.collect()
    
    # Test split: chỉ dùng order trong test_order_ids
    test_df = train_all[train_all["order_id"].isin(test_order_ids)].copy()
    del train_all; gc.collect()
    
    n_prior_orders = len(prior_ids)
    n_train_orders = len(train_order_ids)
    n_test_orders = len(test_order_ids)
    
    print(f"  [DataLoader] Temporal split (train_ratio={train_ratio}):")
    print(f"    Prior orders          : {n_prior_orders:,}")
    print(f"    Train split orders    : {n_train_orders:,}")
    print(f"    Test split orders     : {n_test_orders:,}")
    print(f"    Model build orders    : {n_prior_orders + n_train_orders:,}")
    
    # Tạo test cases: mỗi sản phẩm trong mỗi order test là 1 query
    # ground truth = các sản phẩm còn lại trong cùng order
    print("  [DataLoader] Building test cases ...")
    groups = test_df.groupby("order_id")["product_id"].apply(list)
    test_cases = []
    for prods in groups.values:
        if len(prods) < 2:
            continue  # Bỏ order chỉ có 1 sản phẩm
        prods_set = set(prods)
        for p in prods:
            gt = list(prods_set - {p})
            test_cases.append((p, gt))
    
    print(f"  [DataLoader] Test cases: {len(test_cases):,}")
    print(f"    (seed_product, ground_truth_list) pairs")
    
    return test_cases, model_df
