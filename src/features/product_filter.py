"""
Bộ lọc sản phẩm — loại bỏ non-food departments/aisles khỏi train data.

Sử dụng trong 01_load_data.py để tạo order_products.parquet sạch,
chỉ chứa các sản phẩm thực phẩm (grocery).
"""
from src.config import EXCLUDED_DEPARTMENTS, EXCLUDED_AISLES


def get_excluded_product_ids(products_df):
    """
    Xác định product_id cần loại bỏ dựa trên cấu hình filter.

    Quy tắc: Loại toàn bộ sản phẩm thuộc EXCLUDED_DEPARTMENTS


    Args:
        products_df: DataFrame [product_id, aisle_id, department_id, ...]

    Returns:
        set[int] — các product_id bị loại
    """
    excluded = set()

    # --- Lọc theo department ---
    dept_mask = products_df['department_id'].isin(EXCLUDED_DEPARTMENTS)
    excluded.update(products_df.loc[dept_mask, 'product_id'].tolist())

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

    # Thống kê theo department bị loại
    dept_excluded = products_df[
        products_df['department_id'].isin(EXCLUDED_DEPARTMENTS)
    ]
    for dept_id in sorted(EXCLUDED_DEPARTMENTS):
        count = dept_excluded[dept_excluded['department_id'] == dept_id].shape[0]
        dept_name = products_df[products_df['department_id'] == dept_id]['department'].iloc[0] if count > 0 else "?"
        print(f"      Dept {dept_id} ({dept_name}): {count:,} sản phẩm")