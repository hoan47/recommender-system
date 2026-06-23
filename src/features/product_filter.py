"""
Bộ lọc sản phẩm — loại bỏ non-food departments khỏi train data.

Sử dụng trong 01_load_data.py để tạo order_products.parquet sạch,
chỉ chứa các sản phẩm thực phẩm (grocery).

Lọc dựa trên department name (EXCLUDED_DEPARTMENT_NAMES từ config)
→ map sang department_id qua departments.csv,
không phụ thuộc vào tên department (Anh/Việt) trong products_df.
"""
import pandas as pd

from src.config import EXCLUDED_DEPARTMENT_NAMES, DEPARTMENTS_FILE

# Cache departments_df để tránh đọc file nhiều lần
_departments_cache = None


def _get_excluded_dept_ids():
    """
    Map EXCLUDED_DEPARTMENT_NAMES → department_id qua departments.csv.
    
    Returns:
        set[int] — các department_id bị loại
    """
    global _departments_cache
    if _departments_cache is None:
        _departments_cache = pd.read_csv(DEPARTMENTS_FILE)
    excluded_ids = set(
        _departments_cache[_departments_cache['department'].isin(EXCLUDED_DEPARTMENT_NAMES)]['department_id']
    )
    return excluded_ids


def get_excluded_product_ids(products_df):
    """
    Xác định product_id cần loại bỏ dựa trên department name.

    Quy tắc: Loại toàn bộ sản phẩm có department nằm trong
             EXCLUDED_DEPARTMENT_NAMES (map qua departments.csv để
             lấy department_id, không phụ thuộc tên Anh/Việt).

    Args:
        products_df: DataFrame [product_id, aisle_id, department_id, department, ...]

    Returns:
        set[int] — các product_id bị loại
    """
    excluded_dept_ids = _get_excluded_dept_ids()
    dept_mask = products_df['department_id'].isin(excluded_dept_ids)
    excluded = set(products_df.loc[dept_mask, 'product_id'].tolist())
    return excluded


def filter_order_products(order_products_df, excluded_product_ids):
    """
    Loại bỏ các dòng chứa product_id bị excluded khỏi order_products.

    Args:
        order_products_df: DataFrame [order_id, product_id, ...]
        excluded_product_ids: set[int] — các product_id cần loại

    Returns:
        DataFrame đã lọc
    """
    initial = len(order_products_df)
    filtered = order_products_df[~order_products_df['product_id'].isin(excluded_product_ids)]
    removed = initial - len(filtered)

    return filtered, removed


def get_filter_stats(products_df, order_products_df, excluded_product_ids):
    """
    In thống kê về ảnh hưởng của filter.

    Args:
        products_df: DataFrame products
        order_products_df: DataFrame order_products (sau khi lọc)
        excluded_product_ids: set[int] — các product_id bị loại
    """
    total_products = len(products_df)
    excluded = len(excluded_product_ids)
    kept = total_products - excluded

    print(f"\n  Thống kê Filter Strategy:")
    print(f"    Tổng sản phẩm: {total_products:,}")
    print(f"    Sản phẩm bị loại: {excluded:,} ({excluded/total_products*100:.1f}%)")
    print(f"    Sản phẩm giữ lại: {kept:,} ({kept/total_products*100:.1f}%)")

    # Thống kê theo department bị loại (dùng department_id để không phụ thuộc tên Anh/Việt)
    excluded_dept_ids = _get_excluded_dept_ids()
    excluded_df = products_df[products_df['department_id'].isin(excluded_dept_ids)]
    # Dùng cache thay vì đọc lại DEPARTMENTS_FILE
    dept_id_to_name = dict(zip(_departments_cache['department_id'], _departments_cache['department']))
    for dept_id in sorted(excluded_dept_ids):
        count = excluded_df[excluded_df['department_id'] == dept_id].shape[0]
        dept_name = dept_id_to_name.get(dept_id, "?")
        print(f"      Dept {dept_id} ({dept_name}): {count:,} sản phẩm")