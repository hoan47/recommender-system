# data_loader.py
# Chien luoc su dung du lieu:
#   - order_products_train.csv (80%): Dung de build model (KG, CF, CB)
#   - order_products_test.csv  (20%): Dung de evaluation (test_cases)
#
# Khong con thuc hien split trong code vi du lieu da duoc pre-split.

import pandas as pd
import numpy as np
import gc, os, psutil
from config import (PATH_TRAIN, PATH_TEST, PATH_PRODUCTS,
                    PATH_DEPARTMENTS, MIN_FREQ, MAX_CASES)


def ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

def log(msg):
    print(f"  [RAM {ram_mb():>6.0f} MB]  {msg}")


# ── Bien toan cuc ─────────────────────────────────────────────────────────────
prior_f             = None   # Du lieu train (tu order_products_train.csv)
frequent_items      = None   # set product_id
products            = None
dept_name           = None
dept_ids            = None
prod_dept_map       = None
name_map            = None
user_history        = None
user_dept_pref      = None
MIN_FREQ            = MIN_FREQ

# Tap danh gia
test_cases          = None   # Du lieu test (tu order_products_test.csv)


def load_all():
    global prior_f, frequent_items, products
    global dept_name, dept_ids, prod_dept_map, name_map
    global user_history, user_dept_pref
    global test_cases

    print("\n" + "=" * 65)
    print("LOAD -- Reading pre-split CSV files ...")
    print("=" * 65)

    # ── Doc raw data ───────────────────────────────────────────────────────────
    train_df = pd.read_csv(PATH_TRAIN)
    test_df  = pd.read_csv(PATH_TEST)
    products = pd.read_csv(PATH_PRODUCTS)
    depts    = pd.read_csv(PATH_DEPARTMENTS)

    _cast_tx(train_df); _cast_tx(test_df)
    products['product_id']    = products['product_id'].astype(np.int32)
    products['aisle_id']      = products['aisle_id'].astype(np.int16)
    products['department_id'] = products['department_id'].astype(np.int8)

    dept_name = depts.set_index('department_id')['department'].to_dict()
    dept_ids  = sorted(dept_name.keys())
    has_user  = 'user_id' in train_df.columns

    log(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | Products: {len(products):,}")

    # ── Frequent items (tu train) ──────────────────────────────────────────────
    print("\nFILTER -- Frequent items (from train) ...")
    counts         = train_df['product_id'].value_counts()
    frequent_items = set(counts[counts >= MIN_FREQ].index.astype(np.int32))
    del counts; gc.collect()

    # ── Lookup maps ────────────────────────────────────────────────────────────
    prod_dept_map = {int(k): int(v)
                     for k, v in products.set_index('product_id')['department_id'].items()}
    name_map = products.set_index('product_id')['product_name'].to_dict()

    # ── Chuan bi prior_f (loc frequent) ───────────────────────────────────────
    print("\nFILTER -- Filtering train data by frequent items ...")
    prior_f = train_df[train_df['product_id'].isin(frequent_items)].copy()
    
    # Loc lai de dam bao moi don hang co >= 2 san pham sau khi loc frequent
    mask = prior_f.groupby('order_id', sort=False)['product_id'].transform('count') > 1
    prior_f = prior_f[mask].reset_index(drop=True)
    del train_df, mask; gc.collect()
    log(f"prior_f (train) ready: {len(prior_f):,} rows")

    # ── Chuan bi test_cases (loc frequent) ────────────────────────────────────
    print("\nBUILD TEST -- Building test cases from test file ...")
    test_freq = test_df[test_df['product_id'].isin(frequent_items)].copy()
    
    # Gom nhom theo order_id
    test_grp = (test_freq.sort_values('add_to_cart_order')
                         .groupby('order_id')['product_id']
                         .apply(list))
    
    # Chi lay don hang co >= 2 san pham (1 seed + 1 ground truth)
    test_grp = test_grp[test_grp.apply(len) >= 2]
    
    # Format: [(seed_pid, [ground_truth_pids]), ...]
    test_cases = [(g[0], g[1:]) for g in test_grp][:MAX_CASES]
    
    del test_df, test_freq, test_grp; gc.collect()
    log(f"test_cases ready: {len(test_cases):,} cases")

    # ── User profiles ──────────────────────────────────────────────────────────
    if has_user:
        _build_user_data()
    else:
        user_history   = {}
        user_dept_pref = {}

    log("Data ready.")
    _print_summary()


def _build_user_data():
    global user_history, user_dept_pref
    print("\nPROFILES -- Building user profiles ...")
    srt = prior_f.sort_values(['order_id', 'add_to_cart_order'])
    grp = srt.groupby('user_id', sort=False)['product_id'].apply(list)
    user_history = {int(uid): pids for uid, pids in grp.items()}

    tmp = srt[['user_id', 'product_id']].copy()
    tmp['dept_id'] = tmp['product_id'].map(prod_dept_map)
    tmp = tmp.dropna(subset=['dept_id'])
    dept_g = (tmp.groupby(['user_id', 'dept_id']).size().reset_index(name='cnt'))
    user_dept_pref = {}
    for uid, sub in dept_g.groupby('user_id'):
        user_dept_pref[int(uid)] = dict(zip(sub['dept_id'].astype(int),
                                            sub['cnt'].astype(int)))
    log(f"User profiles: {len(user_history):,} users")


def _cast_tx(df: pd.DataFrame):
    casts = {'order_id': np.int32, 
             'product_id': np.int32,
             'user_id': np.int32, 
             'add_to_cart_order': np.int16,
             'reordered': np.int8}
    for col, dtype in casts.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)


def _print_summary():
    print(f"""
  DATA SUMMARY:
    prior_f (build model) : {len(prior_f):,} rows
    frequent_items        : {len(frequent_items):,} products (>= {MIN_FREQ} purchases)
    test_cases            : {len(test_cases):,} cases
""")
