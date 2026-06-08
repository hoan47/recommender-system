"""
Department direction filter — xác định chiều gợi ý giữa các department

Dựa trên model 1 (RS_Instacart-main/dept_direction.py)

Tính Confidence & Lift theo cặp department, xác định chiều có hướng:
  - Nếu Confidence(A→B) > threshold và Lift > threshold → cho phép gợi ý B từ A
  - Nếu 2 chiều mạnh tương đương → bidirectional
  - Nếu 1 chiều mạnh → directed

Sau đó dùng để filter: chỉ recommend sản phẩm từ department được phép.
"""

import gc
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import permutations
from tqdm import tqdm

from src.config import MIN_CONF, MIN_LIFT, ASYMMETRY_RATIO


# ── Biến toàn cục ────────────────────────────────────────────
dept_suggest = None   # dict: dept_id → set of allowed target dept_ids


def build_dept_direction(prior_df, prod_dept_map):
    """
    Xây dựng bảng direction giữa các department.

    Args:
        prior_df: DataFrame — dữ liệu prior với cột order_id, product_id
        prod_dept_map: dict — product_id → department_id

    Returns:
        dict: dept_id → set of allowed target dept_ids
    """
    global dept_suggest
    print("\n" + "=" * 65)
    print("  DEPT DIRECTION -- Building association rules ...")
    print("=" * 65)

    # ── Basket ở mức department ──────────────────────────────
    print("  Building dept-level baskets ...")
    dept_basket = (
        prior_df[['order_id', 'product_id']]
        .copy()
        .assign(department_id=lambda d: d['product_id'].map(prod_dept_map))
        .dropna(subset=['department_id'])
        [['order_id', 'department_id']]
        .drop_duplicates()
        .astype({'department_id': np.int64})
    )

    total_orders = dept_basket['order_id'].nunique()
    dept_ord_cnt = dept_basket['department_id'].value_counts().to_dict()

    # ── Co-occurrence có hướng ───────────────────────────────
    print("  Counting directed co-occurrences ...")
    dept_groups = dept_basket.groupby('order_id', sort=False)['department_id'].apply(list)
    del dept_basket
    gc.collect()

    pair_counts = defaultdict(int)
    for depts in tqdm(dept_groups, desc="  Dept co-occ", ncols=80):
        uniq = list(set(depts))
        if len(uniq) < 2:
            continue
        for a, b in permutations(uniq, 2):
            pair_counts[(a, b)] += 1

    del dept_groups
    gc.collect()

    # ── Tính Confidence & Lift ───────────────────────────────
    print("  Computing Confidence & Lift ...")
    conf_lookup = {}
    lift_lookup = {}

    for (A, B), co_occ in pair_counts.items():
        if A not in dept_ord_cnt or B not in dept_ord_cnt:
            continue
        p_A = dept_ord_cnt[A] / total_orders
        p_B = dept_ord_cnt[B] / total_orders
        p_AB = co_occ / total_orders
        conf_lookup[(A, B)] = (co_occ / dept_ord_cnt[A]) * 100.0
        lift_lookup[(A, B)] = p_AB / (p_A * p_B + 1e-12)

    del pair_counts
    gc.collect()

    # ── Xác định chiều có hướng ──────────────────────────────
    print("  Determining directions ...")
    dept_suggest = defaultdict(set)
    visited = set()

    for (A, B) in list(conf_lookup.keys()):
        if (A, B) in visited:
            continue
        visited.add((A, B))
        visited.add((B, A))

        conf_AB = conf_lookup.get((A, B), 0.0)
        conf_BA = conf_lookup.get((B, A), 0.0)
        lift_AB = lift_lookup.get((A, B), 0.0)
        lift_BA = lift_lookup.get((B, A), 0.0)

        ab_ok = conf_AB >= MIN_CONF and lift_AB >= MIN_LIFT
        ba_ok = conf_BA >= MIN_CONF and lift_BA >= MIN_LIFT

        if not ab_ok and not ba_ok:
            continue

        if ab_ok and ba_ok:
            ratio = conf_AB / (conf_BA + 1e-9)
            if ratio >= ASYMMETRY_RATIO:
                direction = 'A_to_B'
            elif ratio <= 1.0 / ASYMMETRY_RATIO:
                direction = 'B_to_A'
            else:
                direction = 'bidirectional'
        elif ab_ok:
            direction = 'A_to_B'
        else:
            direction = 'B_to_A'

        if direction in ('A_to_B', 'bidirectional'):
            dept_suggest[A].add(B)
        if direction in ('B_to_A', 'bidirectional'):
            dept_suggest[B].add(A)

    # Cùng department: luôn gợi ý lẫn nhau
    all_depts = list(dept_ord_cnt.keys())
    for did in all_depts:
        dept_suggest[did].add(did)

    # ── In tổng kết ──────────────────────────────────────────
    n_dirs = sum(len(v) - 1 for v in dept_suggest.values())  # trừ self
    print(f"\n  DEPT DIRECTION SUMMARY: {n_dirs} directed pairs found")
    print(f"  (from {len(dept_suggest)} departments)")

    return dept_suggest


def filter_by_direction(seed_pid, candidate_pids, prod_dept_map):
    """
    Giữ lại ứng viên thuộc department được phép gợi ý từ department của seed.

    Args:
        seed_pid: int — sản phẩm gốc
        candidate_pids: list[int] — danh sách ứng viên
        prod_dept_map: dict — product_id → department_id

    Returns:
        list[int] — danh sách đã lọc
    """
    global dept_suggest
    if dept_suggest is None:
        return candidate_pids

    seed_dept = prod_dept_map.get(seed_pid, -1)
    allowed_dept = dept_suggest.get(seed_dept, {seed_dept})

    return [
        p for p in candidate_pids
        if prod_dept_map.get(p, -1) in allowed_dept
    ]