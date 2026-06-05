import pandas as pd
import numpy as np
import gc, os, psutil
from config import (PATH_PRIOR, PATH_TRAIN, PATH_PRODUCTS,
                    PATH_DEPARTMENTS, MIN_FREQ, MAX_CASES)


def ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

def log(msg):
    print(f"  [RAM {ram_mb():>6.0f} MB]  {msg}")


prior_f        = None
frequent_items = None
test_cases     = None
products       = None
dept_name      = None
dept_ids       = None
prod_dept_map  = None
name_map       = None


def load_all():
    global prior_f, frequent_items, test_cases
    global products, dept_name, dept_ids, prod_dept_map, name_map
    global user_history, user_dept_pref

    print("\n" + "=" * 65)
    print("LOAD -- Reading CSV files ...")
    print("=" * 65)

    prior       = pd.read_csv(PATH_PRIOR)
    products    = pd.read_csv(PATH_PRODUCTS)
    departments = pd.read_csv(PATH_DEPARTMENTS)

    _cast_tx(prior)
    products['product_id']    = products['product_id'].astype(np.int32)
    products['aisle_id']      = products['aisle_id'].astype(np.int16)
    products['department_id'] = products['department_id'].astype(np.int8)

    dept_name = departments.set_index('department_id')['department'].to_dict()
    dept_ids  = sorted(dept_name.keys())
    print("\nCLEAN -- Single-item orders ...")
    mask        = prior.groupby('order_id', sort=False)['product_id'].transform('count') > 1
    prior_clean = prior[mask].copy()
    del prior; gc.collect()
    log(f"After clean: {len(prior_clean):,} rows")

    print("\nFILTER -- Frequent items ...")
    counts         = prior_clean['product_id'].value_counts()
    frequent_items = set(counts[counts >= MIN_FREQ].index.astype(np.int32))
    del counts; gc.collect()

    prior_cols = ['order_id', 'product_id', 'reordered', 'add_to_cart_order']

    mask2   = prior_clean['product_id'].isin(frequent_items)
    prior_f_all = prior_clean.loc[mask2, prior_cols].copy()
    mask3   = prior_f_all.groupby('order_id', sort=False)['product_id'].transform('count') > 1
    prior_f = prior_f_all[mask3].reset_index(drop=True)
    del prior_clean, prior_f_all, mask2, mask3; gc.collect()
    log(f"Frequent items: {len(frequent_items):,} | prior_f: {len(prior_f):,}")

    prod_dept_map = {int(k): int(v)
                     for k, v in products.set_index('product_id')['department_id'].items()}
    name_map = products.set_index('product_id')['product_name_vi'].to_dict()


def _cast_tx(df: pd.DataFrame):
    casts = {'order_id': np.int32, 'product_id': np.int32,
             'user_id': np.int32, 'add_to_cart_order': np.int8, 'reordered': np.int8}
    for col, dtype in casts.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)