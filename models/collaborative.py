import math
import gc
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import numpy as np

import data_loader as dl
import dept_direction as dd
from config import KG_SPMI_SHIFT, KG_TOP_SPMI_EDGE, KG_CO_MIN_COUNT

_MAX_BASKET = 15
_CHUNK = 50_000
_PRUNE_MIN = 3
_CROSS_DEPT_BONUS = 1.5
_MAX_PER_DEPT = 3
_SAME_DEPT_MAX = 2

spmi_edges: dict = {}
reorder_rate: dict = {}


def _diversity_filter(seed_pid: int, ranked_pids: list, k: int) -> list:
    seed_dept = dl.prod_dept_map.get(seed_pid, -1)
    dept_count: dict = defaultdict(int)
    result = []
    for pid in ranked_pids:
        if len(result) >= k:
            break
        d = dl.prod_dept_map.get(pid, -1)
        limit = _SAME_DEPT_MAX if d == seed_dept else _MAX_PER_DEPT
        if dept_count[d] < limit:
            dept_count[d] += 1
            result.append(pid)
    return result


def build(prod_dept: dict) -> dict:
    global spmi_edges, reorder_rate

    reorder_rate = {
        int(k): float(v)
        for k, v in dl.train_f.groupby("product_id")["reordered"].mean().items()
    }

    order_grp = (
        dl.train_f[["order_id", "product_id"]]
        .groupby("order_id", sort=False)["product_id"]
        .apply(list)
    )

    co_counts: dict = defaultdict(np.int16)
    oc_counts: dict = defaultdict(np.int32)

    for i, prods in enumerate(tqdm(order_grp, desc="  [CF] Co-occ", ncols=80)):
        uniq = list(set(prods))
        if len(uniq) > _MAX_BASKET:
            continue
        for p in uniq:
            oc_counts[p] += 1
        for a, b in combinations(sorted(uniq), 2):
            co_counts[(a, b)] += 1
        if (i + 1) % _CHUNK == 0:
            co_counts = defaultdict(
                np.int16,
                {k: v for k, v in co_counts.items() if v >= _PRUNE_MIN},
            )
            gc.collect()

    co_counts = {k: int(v) for k, v in co_counts.items() if v >= KG_CO_MIN_COUNT}
    oc_counts = {k: int(v) for k, v in oc_counts.items()}
    gc.collect()

    _build_spmi(co_counts, oc_counts, prod_dept)
    del co_counts, oc_counts
    gc.collect()
    return spmi_edges


def _build_spmi(co_counts: dict, oc_counts: dict, prod_dept: dict) -> None:
    global spmi_edges

    N_est = sum(oc_counts.values()) / 2
    raw: dict = defaultdict(list)

    for (a, b), cnt in tqdm(co_counts.items(), desc="  [CF] SPMI", ncols=80):
        if oc_counts.get(a, 0) == 0 or oc_counts.get(b, 0) == 0:
            continue
        p_ab = cnt / N_est
        p_a = oc_counts[a] / N_est
        p_b = oc_counts[b] / N_est
        pmi = math.log2(p_ab / (p_a * p_b + 1e-12) + 1e-12)
        spmi_val = max(pmi - math.log2(KG_SPMI_SHIFT), 0.0)
        if spmi_val <= 0:
            continue

        rr_bonus = 1.0 + 0.3 * (reorder_rate.get(a, 0) + reorder_rate.get(b, 0)) / 2.0
        dept_a = prod_dept.get(a, -1)
        dept_b = prod_dept.get(b, -1)
        cross = _CROSS_DEPT_BONUS if (dept_a != dept_b and dept_a != -1 and dept_b != -1) else 0.6
        w = spmi_val * rr_bonus * cross

        raw[a].append((b, w))
        raw[b].append((a, w))

    for pid in raw:
        raw[pid].sort(key=lambda x: -x[1])
        spmi_edges[pid] = dict(raw[pid][:KG_TOP_SPMI_EDGE])

    del raw
    gc.collect()
    print(f"  [CF] Done — {len(spmi_edges):,} items")


def recommend(product_id: int, k: int = 100) -> list:
    if product_id not in spmi_edges:
        return []
    scores = spmi_edges[product_id]
    recs = sorted(scores, key=lambda p: -scores[p])
    recs = dd.filter_by_direction(product_id, recs)
    return _diversity_filter(product_id, recs, k)
