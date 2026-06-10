"""
Đọc & merge dữ liệu từ các file CSV gốc.
"""
import pandas as pd
from tqdm import tqdm

from src.config import (
    PRODUCTS_FILE, AISLES_FILE, DEPARTMENTS_FILE,
    ORDER_PRODUCTS_PRIOR, ORDER_PRODUCTS_TRAIN,
    ORDERS_FILE, CHUNKSIZE
)


def load_products() -> pd.DataFrame:
    """
    Đọc products.csv, merge với aisles.csv và departments.csv.

    Returns:
        DataFrame columns: [product_id, product_name, aisle_id,
                           aisle, department_id, department]
    """
    products = pd.read_csv(PRODUCTS_FILE)
    aisles = pd.read_csv(AISLES_FILE)
    departments = pd.read_csv(DEPARTMENTS_FILE)
    
    products = products.merge(aisles, on='aisle_id', how='left')
    products = products.merge(departments, on='department_id', how='left')
    
    return products


def load_order_products(use_prior: bool = True,
                        use_train: bool = True) -> pd.DataFrame:
    """
    Đọc order_products__prior.csv + order_products__train.csv (chunk-based).

    Args:
        use_prior: có đọc file prior không
        use_train: có đọc file train không

    Returns:
        DataFrame columns: [order_id, product_id, add_to_cart_order, reordered]
    """
    chunks = []
    
    if use_prior:
        print("Đang đọc order_products__prior.csv (32.4M records)...")
        for chunk in tqdm(
            pd.read_csv(ORDER_PRODUCTS_PRIOR, chunksize=CHUNKSIZE),
            desc="Prior chunks"
        ):
            chunks.append(chunk)
    
    if use_train:
        print("Đang đọc order_products__train.csv (1.38M records)...")
        for chunk in tqdm(
            pd.read_csv(ORDER_PRODUCTS_TRAIN, chunksize=CHUNKSIZE),
            desc="Train chunks"
        ):
            chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame(columns=['order_id', 'product_id',
                                     'add_to_cart_order', 'reordered'])
    
    df = pd.concat(chunks, ignore_index=True)
    return df


def load_orders(eval_set: str = None) -> pd.DataFrame:
    """
    Đọc orders.csv, lọc theo eval_set nếu có.

    Args:
        eval_set: None → trả về tất cả, hoặc 'prior'/'train'/'test'

    Returns:
        DataFrame columns: [order_id, user_id, eval_set, order_number, ...]
    """
    orders = pd.read_csv(ORDERS_FILE)
    if eval_set is not None:
        orders = orders[orders['eval_set'] == eval_set]
    return orders


def get_product_name_map(products_df: pd.DataFrame = None) -> dict:
    """
    Tạo mapping product_id → product_name.

    Args:
        products_df: DataFrame từ load_products(), nếu None thì tự load

    Returns:
        dict: {product_id: product_name}
    """
    if products_df is None:
        products_df = load_products()
    return dict(zip(products_df['product_id'], products_df['product_name']))