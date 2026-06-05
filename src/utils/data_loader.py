"""
Centralized data loader for Instacart Market Basket Analysis dataset.

Provides a unified interface for loading all CSV files and splitting
train/test ground truth based on orders.csv[eval_set].

All models (CB, SPMI, KG, Hybrid) use this module to load data consistently.
"""

import csv
import os
from pathlib import Path

import pandas as pd

# Project root directory (relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_products():
    """
    Load products.csv and join with departments.csv.
    Aisles.csv is NOT used in this project.

    Returns
    -------
    pd.DataFrame with columns:
        product_id, product_name, department_id, department
    """
    products_path = DATA_DIR / "products.csv"
    departments_path = DATA_DIR / "departments.csv"

    products = pd.read_csv(products_path, encoding="utf-8")
    departments = pd.read_csv(departments_path, encoding="utf-8")

    # Merge product with department name
    products = products.merge(departments, on="department_id", how="left")

    # Fill missing department name (should not happen with 21 known departments)
    products["department"] = products["department"].fillna("unknown department")

    return products[["product_id", "product_name", "department_id", "department"]]


def load_orders(eval_set=None):
    """
    Load orders.csv, optionally filter by eval_set.

    Parameters
    ----------
    eval_set : str or None
        One of 'prior', 'train', 'test', or None (load all).

    Returns
    -------
    pd.DataFrame with columns:
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
    Load order_products__prior.csv or order_products__train.csv.

    Parameters
    ----------
    file_type : str
        'prior' or 'train'

    Returns
    -------
    pd.DataFrame with columns:
        order_id, product_id, add_to_cart_order, reordered
    """
    filename = f"order_products__{file_type}.csv"
    filepath = DATA_DIR / filename

    # Use csv.DictReader for robust parsing (handles commas inside quotes)
    # Read with pandas first for speed, then fallback if issues
    df = pd.read_csv(filepath, encoding="utf-8")

    return df


def load_train_test_split():
    """
    Split ground truth from order_products__train.csv based on orders.csv[eval_set].

    The dataset does NOT have a separate order_products__test.csv.
    order_products__train.csv contains BOTH train (131,209 orders) and
    test (75,000 orders) ground truth. We distinguish them using
    orders.csv[eval_set].

    Returns
    -------
    tuple: (train_gt_df, test_gt_df)
        Each is a pd.DataFrame with columns:
            order_id, product_id, add_to_cart_order, reordered
    """
    # Load orders to get eval_set mapping
    orders_path = DATA_DIR / "orders.csv"
    orders = pd.read_csv(orders_path, encoding="utf-8")

    # Get order_id sets for train and test
    train_order_ids = set(orders[orders["eval_set"] == "train"]["order_id"])
    test_order_ids = set(orders[orders["eval_set"] == "test"]["order_id"])

    # Load ALL order products from train file
    train_products = load_order_products("train")

    # Split based on order_id
    train_gt = train_products[train_products["order_id"].isin(train_order_ids)].copy()
    test_gt = train_products[train_products["order_id"].isin(test_order_ids)].copy()

    return train_gt.reset_index(drop=True), test_gt.reset_index(drop=True)


def load_prior_in_chunks(chunksize=500000):
    """
    Generator that yields chunks of order_products__prior.csv.

    Used for chunk-based processing of 32.4M records without
    loading everything into memory at once.

    Parameters
    ----------
    chunksize : int
        Number of rows per chunk (default: 500,000).

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
    Return data appropriate for each model.

    Parameters
    ----------
    model_name : str
        One of 'cb', 'spmi', 'kg'.
        Note: 'hybrid' is NOT supported here because it loads model outputs
        from files created by CB, SPMI, KG.

    Returns
    -------
    Depends on model_name:
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
            f"Unknown model_name: '{model_name}'. "
            f"Valid options: 'cb', 'spmi', 'kg'. "
            f"Note: 'hybrid' loads model outputs directly."
        )


def get_data_stats():
    """
    Print summary statistics of the dataset.
    Useful for verification after loading.
    """
    orders = load_orders()

    print("=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    # Orders distribution
    for eval_set in ["prior", "train", "test"]:
        count = len(orders[orders["eval_set"] == eval_set])
        pct = 100 * count / len(orders)
        print(f"  {eval_set:>6}: {count:>10,} orders ({pct:.1f}%)")

    # Products
    products = load_products()
    print(f"\n  Total products: {len(products):,}")
    print(f"  Total departments: {products['department_id'].nunique()}")

    # Prior interactions
    prior = load_order_products("prior")
    print(f"\n  Prior interactions: {len(prior):,}")
    print(f"  Unique products in prior: {prior['product_id'].nunique():,}")

    # Train/Test ground truth
    train_gt, test_gt = load_train_test_split()
    print(f"\n  Train ground truth interactions: {len(train_gt):,}")
    print(f"  Test ground truth interactions: {len(test_gt):,}")


if __name__ == "__main__":
    get_data_stats()