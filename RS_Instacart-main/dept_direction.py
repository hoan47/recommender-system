# dept_direction.py
# Xac dinh chieu goi y co huong giua cac danh muc san pham
# dua tren Confidence va Lift cua luat ket hop

import gc
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import permutations
from tqdm import tqdm

import data_loader as dl
from config import (MIN_CONF, MIN_LIFT, ASYMMETRY_RATIO,
                    PATH_DEPT_DIR)


# ── Bien toan cuc ─────────────────────────────────────────────────────────────
dept_suggest  = None   # dict: dept_id -> set of allowed target dept_ids
conf_lookup   = {}     # (A, B) -> confidence A->B
lift_lookup   = {}     # (A, B) -> lift A->B


def build():
    """
    Tinh Confidence & Lift theo cap danh muc, xac dinh chieu co huong.
    Ghi ket qua vao bien toan cuc dept_suggest, conf_lookup, lift_lookup.
    """
    global dept_suggest, conf_lookup, lift_lookup

    print("\n" + "=" * 65)
    print("DEPT DIRECTION -- Building association rules ...")
    print("=" * 65)

    # ── Basket o muc department ───────────────────────────────────────────────
    print("  Building dept-level baskets ...")
    dept_basket = (
        dl.prior_f[['order_id', 'product_id']]
        .copy()
        .assign(department_id=lambda d: d['product_id'].map(dl.prod_dept_map))
        .dropna(subset=['department_id'])
        [['order_id', 'department_id']]
        .drop_duplicates()
        .astype({'department_id': np.int8})
    )

    total_orders   = dept_basket['order_id'].nunique()
    dept_ord_cnt   = dept_basket['department_id'].value_counts().to_dict()

    # ── Co-occurrence co huong ────────────────────────────────────────────────
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

    # ── Tinh Confidence & Lift ────────────────────────────────────────────────
    print("  Computing Confidence & Lift ...")
    conf_lookup = {}
    lift_lookup = {}

    for (A, B), co_occ in pair_counts.items():
        if A not in dept_ord_cnt or B not in dept_ord_cnt:
            continue
        p_A  = dept_ord_cnt[A] / total_orders
        p_B  = dept_ord_cnt[B] / total_orders
        p_AB = co_occ / total_orders
        conf_lookup[(A, B)] = (co_occ / dept_ord_cnt[A]) * 100.0
        lift_lookup[(A, B)] = p_AB / (p_A * p_B + 1e-12)

    del pair_counts
    gc.collect()

    # ── Xac dinh chieu co huong ───────────────────────────────────────────────
    print("  Determining directions ...")
    dept_suggest = defaultdict(set)
    visited      = set()
    direction_log = []

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

        direction_log.append({
            'from'     : dl.dept_name.get(A, str(A)),
            'to'       : dl.dept_name.get(B, str(B)),
            'conf_A_B' : round(conf_AB, 2),
            'conf_B_A' : round(conf_BA, 2),
            'lift_A_B' : round(lift_AB, 3),
            'direction': direction,
        })

    # Cung danh muc: luon goi y nhau
    for did in dl.dept_ids:
        dept_suggest[did].add(did)

    # ── In tong ket ───────────────────────────────────────────────────────────
    print("\n  DEPT DIRECTION SUMMARY:")
    print(f"  {'Source':<20} -> Allowed targets")
    print("  " + "-" * 60)
    for did in dl.dept_ids:
        targets = [dl.dept_name.get(t, str(t))
                   for t in sorted(dept_suggest.get(did, set())) if t != did]
        if targets:
            print(f"  {dl.dept_name.get(did,'?'):<20} -> {', '.join(targets)}")

    # Luu CSV
    pd.DataFrame(direction_log).to_csv(PATH_DEPT_DIR, index=False,
                                       encoding='utf-8-sig')
    print(f"\n  Saved -> {PATH_DEPT_DIR}")


def filter_by_direction(seed_pid: int, candidate_pids: list) -> list:
    """
    Giu lai ung vien thuoc danh muc duoc phep goi y tu danh muc cua seed.
    Cung danh muc voi seed luon duoc giu lai.
    Neu dept_suggest chua duoc build (None), tra ve toan bo candidates (khong loc).
    """
    if dept_suggest is None:
        return candidate_pids
    seed_dept    = dl.prod_dept_map.get(seed_pid, -1)
    allowed_dept = dept_suggest.get(seed_dept, {seed_dept})
    return [p for p in candidate_pids
            if dl.prod_dept_map.get(p, -1) in allowed_dept]