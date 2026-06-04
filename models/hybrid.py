from collections import defaultdict

import data_loader as dl
import dept_direction as dd
from config import CB_FILTER_HIGH, HYB_ALPHA
from . import collaborative, knowledge_graph, content_based

_MAX_PER_DEPT = 3
_SAME_DEPT_MAX = 2


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


def recommend(product_id: int, k: int = 100) -> list:
    in_graph = product_id in knowledge_graph.graph_nb
    in_spmi = product_id in collaborative.spmi_edges
    if not in_graph and not in_spmi:
        return []

    cf_scores = collaborative.spmi_edges.get(product_id, {})
    max_cf = max(cf_scores.values()) if cf_scores else 1.0

    kg_scores = knowledge_graph._rwr(product_id)
    max_kg = max(kg_scores.values()) if kg_scores else 1.0

    pool = set(cf_scores.keys()) | set(kg_scores.keys())

    hybrid_scores: dict = {}
    for pid in pool:
        if pid == product_id or pid not in dl.frequent_items:
            continue
        if content_based.similarity(product_id, pid) >= CB_FILTER_HIGH:
            continue
        dept_a = dl.prod_dept_map.get(product_id, -1)
        dept_b = dl.prod_dept_map.get(pid, -1)
        dept_mult = 1.4 if (dept_a != dept_b and dept_a != -1 and dept_b != -1) else 0.5
        norm_cf = cf_scores.get(pid, 0.0) / (max_cf + 1e-9)
        norm_kg = kg_scores.get(pid, 0.0) / (max_kg + 1e-9)
        hybrid_scores[pid] = (HYB_ALPHA * norm_cf + (1.0 - HYB_ALPHA) * norm_kg) * dept_mult

    recs = sorted(hybrid_scores, key=lambda p: -hybrid_scores[p])
    recs = dd.filter_by_direction(product_id, recs)
    return _diversity_filter(product_id, recs, k)
