"""
Unified Scoring — Confidence Item-based Collaborative Filtering

Công thức unified scoring (asymmetric, directed graph):
  ochiai(A,B) = cnt / sqrt(freq[A] * freq[B])          # Cosine Similarity
  log_ab      = log1p(cnt)                               # Popularity Bonus (log)
  conf(A→B)   = cnt / freq[A]                            # Conditional Probability
  score(A→B)  = ochiai(A,B) * conf(A→B) * log_ab        # Unified score

  Reorder bonus:
    rr_bonus = 1.0 + REORDER_BONUS * avg(reorder_rate[A], reorder_rate[B])
    score *= rr_bonus

  Cross-dept bonus:
    Nếu A và B khác department → cross_bonus > 1 (ưu tiên mua kèm)
    Cùng department → cross_bonus < 1 (giảm substitute)

Kĩ thuật:
  - Đếm co-occurrence bằng defaultdict + combinations (Python thuần, không Numba)
  - Lưu kết quả dạng dict[pid, dict[npid, score]] — đơn giản, dễ đọc
  - Chunking mỗi 50K order để kiểm soát bộ nhớ
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import math
import json
import numpy as np
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

from src.config import MODELS_DIR, CONF_FREQ_MIN, CONF_TOP_K, REORDER_BONUS

# File lưu co_occ_edges dạng json
CONF_FILE = MODELS_DIR / "confidence_dict.json"

# Biến toàn cục: {pid: {npid: score}}
# score = ochiai * conf * log1p * rr_bonus * cross_bonus
co_occ_edges = {}

# Cache department: {pid: dept_id} — được set từ ngoài vào
_prod_dept = {}

# Hằng số
_MAX_BASKET = 15
_CHUNK = 50_000
_PRUNE_MIN = 3
_CROSS_DEPT_BONUS = 1.3


def set_dept_map(dept_map):
    """Nhận department map từ ngoài: dict[pid, dept_id]"""
    global _prod_dept
    _prod_dept = dept_map


def _get_dept(pid):
    return _prod_dept.get(pid, -1)


def build_cooc(prior_df):
    """
    Đếm co-occurrence từ prior orders.
    Dùng defaultdict + combinations — Python thuần, dễ hiểu.

    Tham số:
        prior_df: DataFrame — cần cột order_id, product_id

    Trả về:
        co_counts: dict[(a,b), count] — số lần a và b xuất hiện cùng order
        oc_counts: dict[pid, count] — số order chứa sản phẩm đó
    """
    print("\n  [Confidence] Building co-occurrence ...")

    # Nhóm sản phẩm theo từng order
    order_grp = (prior_df[['order_id', 'product_id']]
                   .groupby('order_id', sort=False)['product_id'].apply(list))

    co_counts = defaultdict(np.int16)
    oc_counts = defaultdict(np.int32)

    for i, prods in enumerate(tqdm(order_grp, desc="  Co-occ", ncols=80)):
        uniq = list(set(prods))
        if len(uniq) > _MAX_BASKET:
            continue  # Bỏ qua order quá lớn (nhiễu)

        # Đếm tần suất từng sản phẩm
        for p in uniq:
            oc_counts[p] += 1

        # Đếm co-occurrence cho từng cặp
        for a, b in combinations(sorted(uniq), 2):
            co_counts[(a, b)] += 1

        # Prune định kỳ để kiểm soát bộ nhớ
        if (i + 1) % _CHUNK == 0:
            co_counts = defaultdict(
                np.int16,
                {k: v for k, v in co_counts.items() if v >= _PRUNE_MIN}
            )
            gc.collect()

    # Lọc lần cuối
    co_counts = {k: int(v) for k, v in co_counts.items() if v >= CONF_FREQ_MIN}
    oc_counts = {k: int(v) for k, v in oc_counts.items()}

    gc.collect()
    print(f"  [Confidence] {len(co_counts):,} co-occur pairs, "
          f"{len(oc_counts):,} products")
    return co_counts, oc_counts


def build_confidence(co_counts, oc_counts, reorder_rate=None):
    """
    Tính unified score cho mỗi cặp sản phẩm, lưu vào co_occ_edges.

    Công thức:
      ochiai = cnt / sqrt(freq[i] * freq[j])
      log_ab = log1p(cnt)
      conf   = cnt / freq[i]
      score  = ochiai * conf * log_ab * rr_bonus * cross_bonus

    Chỉ giữ TOP_K sản phẩm có score cao nhất cho mỗi sản phẩm.
    """
    global co_occ_edges
    print("\n  [Confidence] Computing unified scores ...")

    # Gom score theo source product
    temp = defaultdict(list)

    for (a, b), cnt in tqdm(co_counts.items(), desc="  Scoring", ncols=80):
        if oc_counts.get(a, 0) == 0 or oc_counts.get(b, 0) == 0:
            continue

        # Thành phần chung
        ochiai = cnt / math.sqrt(oc_counts[a] * oc_counts[b])
        log_ab = math.log1p(cnt)

        # Reorder bonus
        if reorder_rate is not None:
            rr_bonus = 1.0 + REORDER_BONUS * (
                reorder_rate.get(a, 0.5) + reorder_rate.get(b, 0.5)
            ) / 2.0
        else:
            rr_bonus = 1.0

        # Cross-dept bonus
        dept_a = _get_dept(a)
        dept_b = _get_dept(b)
        if dept_a != dept_b and dept_a != -1 and dept_b != -1:
            cross_bonus = _CROSS_DEPT_BONUS
        else:
            cross_bonus = 0.8

        # Score 2 chiều
        conf_a_b = cnt / oc_counts[a]
        score_a_b = ochiai * conf_a_b * log_ab * rr_bonus * cross_bonus
        temp[a].append((b, score_a_b))

        conf_b_a = cnt / oc_counts[b]
        score_b_a = ochiai * conf_b_a * log_ab * rr_bonus * cross_bonus
        temp[b].append((a, score_b_a))

    # Chỉ giữ TOP_K cho mỗi sản phẩm
    for pid, scores in tqdm(temp.items(), desc="  Top-K", ncols=80):
        scores.sort(key=lambda x: -x[1])
        co_occ_edges[pid] = {npid: s for npid, s in scores[:CONF_TOP_K]}

    del temp, co_counts, oc_counts
    gc.collect()
    print(f"  [Confidence] Done: {len(co_occ_edges):,} products with edges")


def save():
    """Lưu co_occ_edges ra file json (key str vì json không hỗ trợ int key)"""
    out = {}
    for pid, edges in co_occ_edges.items():
        out[str(pid)] = {str(npid): float(s) for npid, s in edges.items()}
    with open(CONF_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  [Confidence] Saved: {CONF_FILE}")


def load():
    """Load co_occ_edges từ file json, chuyển key từ str về int"""
    global co_occ_edges
    with open(CONF_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    co_occ_edges = {
        int(pid): {int(npid): float(s) for npid, s in edges.items()}
        for pid, edges in raw.items()
    }
    print(f"  [Confidence] Loaded: {len(co_occ_edges):,} products")


# ===== Hàm tiện ích: recommend từ co_occ_edges =====
def recommend(seed_pid, top_k=100):
    """Trả về danh sách product_id được gợi ý cho seed, giảm dần theo score"""
    edges = co_occ_edges.get(seed_pid)
    if not edges:
        return []
    sorted_pids = sorted(edges, key=lambda p: -edges[p])
    return sorted_pids[:top_k]


if __name__ == "__main__":
    from src.data_loader import load_prior, load_products

    prior = load_prior()
    products_df = load_products()

    # Map department
    dept_map = dict(zip(products_df["product_id"], products_df["department_id"]))
    set_dept_map(dept_map)

    # Tính reorder rate
    reorder_rate = prior.groupby('product_id')['reordered'].mean().to_dict()

    co_counts, oc_counts = build_cooc(prior)
    build_confidence(co_counts, oc_counts, reorder_rate=reorder_rate)
    save()