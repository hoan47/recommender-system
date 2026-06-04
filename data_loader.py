import pandas as pd
import numpy as np
import gc
import os
import psutil
from config import (
    PATH_TRAIN, PATH_PRODUCTS,
    PATH_DEPARTMENTS, MIN_FREQ, MAX_CASES,
)

train_f        = None # Từ nguồn train.csv
frequent_items = None # Lọc từ train_f, product_id xuất hiện ≥ MIN_FREQ
test_cases     = None # (seed, ground_truth) seed là sản phẩm đầu tiên của đơn, ground_truth là list sản phẩm tiếp theo của đơn đó
products       = None
dept_name      = None
dept_ids       = None
prod_dept_map  = None
name_map       = None


def _ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _log(msg: str) -> None:
    print(f"  [RAM {_ram_mb():>6.0f} MB]  {msg}")


def load_all() -> None:
    global train_f, frequent_items, test_cases
    global products, dept_name, dept_ids, prod_dept_map, name_map

    print("\n" + "=" * 65)
    print("LOAD — Reading CSV files")
    print("=" * 65)

    train       = pd.read_csv(PATH_TRAIN)
    products    = pd.read_csv(PATH_PRODUCTS)
    departments = pd.read_csv(PATH_DEPARTMENTS)

    # Cast types tối ưu lưu trữ trong RAM
    _cast_types(train)
    _cast_types(products)
    _cast_types(departments)

    # Tạo mapping từ department_id đến department_name
    dept_name = departments.set_index("department_id")["department"].to_dict()
    dept_ids  = sorted(dept_name.keys())

    _log(f"train {len(train):,} | products {len(products):,}")

    # Chọn đơn hàng có ≥ 2 sản phẩm. Trả về series boolean (series có index khác array)
    mask        = train.groupby("order_id", sort=False)["product_id"].transform("count") > 1
    train_clean = train[mask].copy()
    
    del train
    gc.collect()
    _log(f"After clean: {len(train_clean):,} rows")
    
    # Trả về series
    counts         = train_clean["product_id"].value_counts()
    # Lấy product_id có tầng xuất xuất hiện >= MIN_FREQ xong lưu vào set (hasttable O(1)) để dùng trong việc truy vấn
    frequent_items = set(counts[counts >= MIN_FREQ].index.astype(np.int32))
    del counts
    gc.collect()

    train_cols = [
        "order_id", "user_id", "product_id", "reordered", "add_to_cart_order",
    ]
    # Trả về series boolean
    mask2       = train_clean["product_id"].isin(frequent_items)
    train_f_all = train_clean.loc[mask2, train_cols].copy()
    # Mask3 này không khác gì Mask1 mục đích lọc lại vì trong quá trình ở Mask2 nó đã xóa những product đủ chỉ tiêu ở mask1
    mask3       = train_f_all.groupby("order_id", sort=False)["product_id"].transform("count") > 1
    # reset_index làm lại index mới vì trước đó qua 3 bước mask đã xóa làm khuyết nhiều index
    train_f     = train_f_all[mask3].reset_index(drop=True)
    del train_clean, train_f_all, mask2, mask3
    gc.collect()
    _log(f"Frequent items: {len(frequent_items):,} | train_f: {len(train_f):,}")

    prod_dept_map = {
        int(k): int(v)
        for k, v in products.set_index("product_id")["department_id"].items()
    }
    name_map = products.set_index("product_id")["product_name_vi"].to_dict()

    _build_user_split()

    _log("Data ready.")


def _build_user_split() -> None:
    global train_f, test_cases

    order_user = (
        train_f[["order_id", "user_id"]]
        .drop_duplicates()
        .sort_values("order_id")
    )

    user_order_counts = order_user.groupby("user_id")["order_id"].count()
    active_users      = user_order_counts[user_order_counts >= 5].index

    test_order_ids = []
    for uid in active_users:
        orders  = order_user[order_user["user_id"] == uid]["order_id"].tolist()
        n_test  = max(1, int(len(orders) * 0.2))
        test_order_ids.extend(orders[-n_test:])

    test_order_ids_set = set(test_order_ids)
    train_f = train_f[~train_f["order_id"].isin(test_order_ids_set)].copy()
    gc.collect()

    train_full = pd.read_csv(PATH_TRAIN)
    _cast_types(train_full)
    test_df = train_full[
        train_full["order_id"].isin(test_order_ids_set)
        & train_full["product_id"].isin(frequent_items)
    ].copy()
    del train_full
    gc.collect()

    mask    = test_df.groupby("order_id")["product_id"].transform("count") > 1
    test_df = test_df[mask]

    groups = (
        test_df.sort_values("add_to_cart_order")
        .groupby("order_id")["product_id"]
        .apply(list)
    )
    groups     = groups[groups.apply(len) >= 2]
    test_cases = [(g[0], g[1:]) for g in groups][:MAX_CASES]

    _log(
        f"Test cases (user-split): {len(test_cases):,} | "
        f"test orders: {len(test_order_ids):,} | "
        f"train_f: {len(train_f):,} rows"
    )


# Column -> dtype for any loaded CSV; only columns present in df are cast.
_DTYPE_CASTS = {
    "order_id":          np.int32,
    "product_id":        np.int32,
    "user_id":           np.int32,
    "aisle_id":          np.int16,
    "department_id":     np.int8,
    "add_to_cart_order": np.int8,
    "reordered":         np.int8,
}


def _cast_types(df: pd.DataFrame) -> None:
    for col, dtype in _DTYPE_CASTS.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
