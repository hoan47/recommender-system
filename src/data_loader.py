"""Load dữ liệu Instacart — products, prior, train/test split"""
import gc, pandas as pd
from src.config import DATA_DIR

CASTS = {
    'order_id': 'int32', 'product_id': 'int32', 'user_id': 'int32',
    'add_to_cart_order': 'int8', 'reordered': 'int8',
}

def _cast(df):
    for c, t in CASTS.items():
        if c in df.columns:
            df[c] = df[c].astype(t)
    return df

def load_products():
    """Trả về DataFrame: product_id, product_name, department_id, department"""
    p = pd.read_csv(DATA_DIR / "products.csv", encoding="utf-8")
    d = pd.read_csv(DATA_DIR / "departments.csv", encoding="utf-8")
    _cast(p); _cast(d)
    m = p.merge(d, on="department_id", how="left")
    m["product_name"] = m["product_name"].fillna("unknown")
    m["department"] = m["department"].fillna("unknown")
    gc.collect()
    return m[["product_id", "product_name", "department_id", "department"]]

def load_prior():
    """Trả về DataFrame prior interactions"""
    df = pd.read_csv(DATA_DIR / "order_products__prior.csv", encoding="utf-8")
    _cast(df)
    return df

def load_train_test():
    """
    Tách train/test từ order_products__train.csv dựa trên orders.csv[eval_set].
    Trả về (train_df, test_df).
    """
    orders = pd.read_csv(DATA_DIR / "orders.csv", encoding="utf-8")
    _cast(orders)
    train_ids = orders[orders["eval_set"] == "train"]["order_id"]
    test_ids = orders[orders["eval_set"] == "test"]["order_id"]
    del orders; gc.collect()
    t = pd.read_csv(DATA_DIR / "order_products__train.csv", encoding="utf-8")
    _cast(t)
    train = t[t["order_id"].isin(train_ids)].copy()
    test = t[t["order_id"].isin(test_ids)].copy()
    del t; gc.collect()
    return train, test