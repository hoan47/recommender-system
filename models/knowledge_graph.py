import gc
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import data_loader as dl
import dept_direction as dd
from config import KG_RWR_STEPS, KG_RWR_RESTART, KG_RWR_WALKS

_MAX_PER_DEPT = 3
_SAME_DEPT_MAX = 2

graph_nb: dict = {}
graph_wt: dict = {}


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


def build(spmi_edges: dict) -> None:
    global graph_nb, graph_wt

    _nb: dict = defaultdict(list)
    _wt: dict = defaultdict(list)

    for pid, nbrs in tqdm(spmi_edges.items(), desc="  [KG] Building graph", ncols=80):
        for npid, w in nbrs.items():
            _nb[pid].append(npid)
            _wt[pid].append(float(w))

    for node in list(_nb.keys()):
        wt = np.array(_wt[node], dtype=np.float32)
        wt /= wt.sum() + 1e-9
        graph_nb[node] = np.array(_nb[node], dtype=np.int32)
        graph_wt[node] = wt

    del _nb, _wt
    gc.collect()

    n_nodes = len(graph_nb)
    n_edges = sum(len(v) for v in graph_nb.values())
    print(f"  [KG] Done — {n_nodes:,} nodes | {n_edges:,} edges")


def _rwr(seed: int) -> dict:
    visit: dict = defaultdict(float)
    if seed not in graph_nb:
        return visit
    for _ in range(KG_RWR_WALKS):
        node = seed
        for hop in range(KG_RWR_STEPS):
            if np.random.random() < KG_RWR_RESTART:
                break
            if node not in graph_nb:
                break
            node = int(np.random.choice(graph_nb[node], p=graph_wt[node]))
            if node > 0 and node != seed:
                visit[node] += 1.0 / (hop + 1)
    return visit


def recommend(product_id: int, k: int = 100) -> list:
    if product_id not in graph_nb:
        return []
    scores = _rwr(product_id)
    recs = sorted(
        (p for p in scores if p in dl.frequent_items and p != product_id),
        key=lambda p: -scores[p],
    )
    recs = dd.filter_by_direction(product_id, recs)
    return _diversity_filter(product_id, recs, k)
