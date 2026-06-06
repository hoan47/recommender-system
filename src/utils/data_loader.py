"""
Trình tải dữ liệu tập trung cho dataset Instacart Market Basket Analysis.

Cung cấp giao diện thống nhất để tải tất cả file CSV và tách
ground truth train/test dựa trên orders.csv[eval_set].

Tất cả model (CB, SPMI, KG, Hybrid) dùng module này để tải dữ liệu nhất quán.
"""

import csv
import os

import pandas as pd

from src.config import DATA_DIR, DATA_ENCODING


def load_products():
    """
    Tải products.csv và join với departments.csv.
    File aisles.csv KHÔNG được dùng trong dự án này.

    Trả về
    -------
    pd.DataFrame với các cột:
        product_id, product_name, department_id, department
    """
    products_path = DATA_DIR / "products.csv"
    departments_path = DATA_DIR / "departments.csv"

    products = pd.read_csv(products_path, encoding="utf-8")
    departments = pd.read_csv(departments_path, encoding="utf-8")

    # Merge product với tên department
    products = products.merge(departments, on="department_id", how="left")

    # Điền department bị thiếu (không nên xảy ra với 21 department đã biết)
    products["department"] = products["department"].fillna("unknown department")

    return products[["product_id", "product_name", "department_id", "department"]]


def load_orders(eval_set=None):
    """
    Tải orders.csv, tùy chọn lọc theo eval_set.

    Tham số
    ----------
    eval_set : str hoặc None
        Một trong 'prior', 'train', 'test', hoặc None (tải tất cả).

    Trả về
    -------
    pd.DataFrame với các cột:
        order_id, user_id, eval_set, order_number, order_dow,
        order_hour_of_day, days_since_prior_order
    """
    orders_path = DATA_DIR / "orders.csv"
    orders = pd.read_csv(orders_path, encoding="utf-8")

    if eval_set is not None:
        orders = orders[orders["eval_set"] == eval_set].copy()

    return orders.reset_index(drop=True)


def load_order_products(file_type="prior"):
    """
    Tải order_products__prior.csv hoặc order_products__train.csv.

    Tham số
    ----------
    file_type : str
        'prior' hoặc 'train'

    Trả về
    -------
    pd.DataFrame với các cột:
        order_id, product_id, add_to_cart_order, reordered
    """
    filename = f"order_products__{file_type}.csv"
    filepath = DATA_DIR / filename

    # Dùng pandas để tải nhanh
    df = pd.read_csv(filepath, encoding="utf-8")

    return df


def load_train_test_split():
    """
    Tách ground truth từ order_products__train.csv dựa trên orders.csv[eval_set].

    Dataset KHÔNG có file order_products__test.csv riêng.
    order_products__train.csv chứa CẢ ground truth train (131,209 đơn) và
    test (75,000 đơn). Ta phân biệt chúng qua orders.csv[eval_set].

    Trả về
    -------
    tuple: (train_gt_df, test_gt_df)
        Mỗi cái là pd.DataFrame với các cột:
            order_id, product_id, add_to_cart_order, reordered
    """
    # Tải orders để lấy ánh xạ eval_set
    orders_path = DATA_DIR / "orders.csv"
    orders = pd.read_csv(orders_path, encoding="utf-8")

    # Lấy tập order_id cho train và test
    train_order_ids = set(orders[orders["eval_set"] == "train"]["order_id"])
    test_order_ids = set(orders[orders["eval_set"] == "test"]["order_id"])

    # Tải TẤT CẢ order products từ file train
    train_products = load_order_products("train")

    # Tách dựa trên order_id
    train_gt = train_products[train_products["order_id"].isin(train_order_ids)].copy()
    test_gt = train_products[train_products["order_id"].isin(test_order_ids)].copy()

    return train_gt.reset_index(drop=True), test_gt.reset_index(drop=True)


def load_prior_in_chunks(chunksize=500000):
    """
    Generator trả về từng chunk của order_products__prior.csv.

    Dùng để xử lý 32.4M records theo chunk mà không cần tải
    toàn bộ vào RAM cùng lúc.

    Tham số
    ----------
    chunksize : int
        Số dòng mỗi chunk (mặc định: 500,000).

    Yields
    ------
    pd.DataFrame chunks
    """
    filepath = DATA_DIR / "order_products__prior.csv"
    reader = pd.read_csv(filepath, encoding="utf-8", chunksize=chunksize)
    for chunk in reader:
        yield chunk


def load_data_for_model(model_name):
    """
    Trả về dữ liệu phù hợp cho từng model.

    Tham số
    ----------
    model_name : str
        Một trong 'cb', 'spmi', 'kg'.
        Lưu ý: 'hybrid' KHÔNG được hỗ trợ ở đây vì nó tải model outputs
        từ file do CB, SPMI, KG tạo ra.

    Trả về
    -------
    Tùy theo model_name:
        - 'cb': products_df
        - 'spmi': (prior_products_df, train_gt_df, test_gt_df)
        - 'kg': (prior_products_df, products_df)
    """
    model_name = model_name.lower()

    if model_name == "cb":
        return load_products()

    elif model_name == "spmi":
        prior_products = load_order_products("prior")
        train_gt, test_gt = load_train_test_split()
        return prior_products, train_gt, test_gt

    elif model_name == "kg":
        prior_products = load_order_products("prior")
        products = load_products()
        return prior_products, products

    else:
        raise ValueError(
            f"model_name không hợp lệ: '{model_name}'. "
            f"Giá trị hợp lệ: 'cb', 'spmi', 'kg'. "
            f"Lưu ý: 'hybrid' tải model outputs trực tiếp."
        )


def get_data_stats():
    """
    In thống kê tóm tắt của dataset.
    Hữu ích để kiểm tra sau khi tải.
    """
    orders = load_orders()

    print("=" * 50)
    print("Thống kê Dataset")
    print("=" * 50)

    # Phân bố orders
    for eval_set in ["prior", "train", "test"]:
        count = len(orders[orders["eval_set"] == eval_set])
        pct = 100 * count / len(orders)
        print(f"  {eval_set:>6}: {count:>10,} đơn ({pct:.1f}%)")

    # Products
    products = load_products()
    print(f"\n  Tổng sản phẩm: {len(products):,}")
    print(f"  Tổng department: {products['department_id'].nunique()}")

    # Prior interactions
    prior = load_order_products("prior")
    print(f"\n  Prior interactions: {len(prior):,}")
    print(f"  Sản phẩm unique trong prior: {prior['product_id'].nunique():,}")

    # Train/Test ground truth
    train_gt, test_gt = load_train_test_split()
    print(f"\n  Train ground truth interactions: {len(train_gt):,}")
    print(f"  Test ground truth interactions: {len(test_gt):,}")


if __name__ == "__main__":
    get_data_stats()