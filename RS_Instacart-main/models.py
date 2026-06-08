# models.py
# 4 models: Content-Based, Collaborative (CoOcc), Knowledge Graph (RWR), Hybrid
# Khong dung thu vien ML co san (Word2Vec/Sklearn).
#
# FIX CHINH: Goi y san pham MUA KEM (complementary), KHONG phai san pham tuong tu
#   [1] CoOcc cross-dept bonus: uu tien cap khac danh muc
#   [2] KG: loai bo canh P->Aisle/Dept de tranh keo RWR ve cung dept
#   [3] Cross-dept diversity filter: dam bao ket qua co nhieu danh muc
#   [4] CB dung de loc sản pham GIONG (thay the), khong dung de goi y chinh

import math
import gc
import re
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from tqdm import tqdm

import data_loader as dl
import dept_direction as dd
from config import (
    KG_TOP_EDGE_LIMIT,
    KG_RWR_STEPS, KG_RWR_RESTART, KG_RWR_WALKS,
    KG_CO_MIN_COUNT,
    CB_MAX_DF, CB_MIN_DF, CB_FILTER_HIGH, HYB_ALPHA
)

_MAX_BASKET = 15
_CHUNK      = 50_000
_PRUNE_MIN  = 3

# ── Bien toan cuc ─────────────────────────────────────────────────────────────
cb_vectors        = {}
co_occ_edges_global = {}
graph_nb          = {}
graph_wt          = {}
reorder_rate      = {}
_prod_dept        = {}   # cache: product_id -> dept_id


def _get_dept(pid: int) -> int:
    return _prod_dept.get(pid, -1)


# =============================================================================
#  BUOC 0 -- Tien xu ly dept map
# =============================================================================
def _init_dept_map():
    global _prod_dept
    _prod_dept = {int(k): int(v) for k, v in dl.prod_dept_map.items()}


# =============================================================================
#  BUOC 1 -- Content-Based (Custom TF-IDF)
#  Chu y: CB dung de LOC san pham tuong tu (thay the), KHONG dung de goi y chinh
# =============================================================================
def _build_cb_vectors():
    global cb_vectors
    print("\n" + "=" * 65)
    print("  [Build] Content-Based vectors ...")
    print("=" * 65)

    prod_info = dl.products.set_index('product_id')
    pids = [pid for pid in dl.frequent_items if pid in prod_info.index]

    doc_tokens = {}
    doc_freq   = Counter()
    
    # Danh sách stop words cơ bản tiếng Anh để loại bỏ các từ vô nghĩa
    STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'in', 'on', 'with', 'for', 'of', 'to', 'is', 'at', 'by', 'from', 'with', 'flavored', 'pack', 'oz'}

    for pid in tqdm(pids, desc="  Tokenizing", ncols=80):
        raw  = str(prod_info.loc[pid, 'product_name'])
        text = re.sub(r'[^a-z0-9 ]', ' ', raw.lower()).strip()
        tokens = [t for t in text.split() if t not in STOPWORDS]
        
        ngrams = list(tokens)
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]} {tokens[i+1]}")
            
        doc_tokens[pid] = ngrams
        for t in set(ngrams):
            doc_freq[t] += 1

    n_docs = len(pids)
    vocab  = {}
    idx    = 0
    for t, df in sorted(doc_freq.items(), key=lambda x: -x[1]):
        if df >= CB_MIN_DF and (df / n_docs) <= CB_MAX_DF:
            vocab[t] = idx
            idx += 1
            if idx >= 30_000:
                break

    idf = {t: math.log((1 + n_docs) / (1 + doc_freq[t])) + 1.0
           for t in vocab}

    for pid in tqdm(pids, desc="  TF-IDF", ncols=80):
        tokens = doc_tokens[pid]
        counts = Counter(tokens)
        vec    = {}
        norm_sq = 0.0
        for t, tf in counts.items():
            if t not in vocab:
                continue
            val = (1.0 + math.log(tf)) * idf[t]
            vec[vocab[t]] = val
            norm_sq += val * val
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for t_idx in vec:
                vec[t_idx] /= norm
        cb_vectors[pid] = vec

    del doc_tokens, doc_freq, vocab, idf, prod_info
    gc.collect()
    print(f"  [OK] CB vectors: {len(cb_vectors):,}")


def _cb_similarity(pid_a: int, pid_b: int) -> float:
    va = cb_vectors.get(pid_a)
    vb = cb_vectors.get(pid_b)
    if not va or not vb:
        return 0.0
    if len(va) > len(vb):
        va, vb = vb, va
    return sum(va[t] * vb[t] for t in va if t in vb)


# =============================================================================
#  BUOC 2 -- Co-occurrence + CoOcc (co cross-dept bonus)
#
#  FIX: Them cross-dept bonus vao CoOcc
#    - Cap cung dept (substitutes): CoOcc giu nguyen
#    - Cap khac dept (complements): CoOcc nhan them bonus
#      cross_bonus = 1 + CROSS_DEPT_BONUS * (1 - same_dept_penalty)
#  => Gop cap khac dept len cao hon trong xep hang
# =============================================================================
CROSS_DEPT_BONUS = 1.3   # Giam tu 1.5 xuong 1.3

def _build_stats_and_cooccurrence():
    global reorder_rate
    print("\n  [Build] Co-occurrence (CoOcc + cross-dept bonus) ...")

    reorder_rate = {
        int(k): float(v)
        for k, v in dl.prior_f.groupby('product_id')['reordered'].mean().items()
    }

    order_grp = (dl.prior_f[['order_id', 'product_id']]
                   .groupby('order_id', sort=False)['product_id'].apply(list))

    co_counts = defaultdict(np.int16)
    oc_counts = defaultdict(np.int32)

    for i, prods in enumerate(tqdm(order_grp, desc="  Co-occ", ncols=80)):
        uniq = list(set(prods))
        if len(uniq) > _MAX_BASKET:
            continue
        for p in uniq:
            oc_counts[p] += 1
        for a, b in combinations(sorted(uniq), 2):
            co_counts[(a, b)] += 1
        if (i + 1) % _CHUNK == 0:
            co_counts = defaultdict(np.int16,
                                    {k: v for k, v in co_counts.items()
                                     if v >= _PRUNE_MIN})
            gc.collect()

    co_counts = {k: int(v) for k, v in co_counts.items() if v >= KG_CO_MIN_COUNT}
    oc_counts = {k: int(v) for k, v in oc_counts.items()}
    gc.collect()
    return co_counts, oc_counts


def _build_spmi_edges(co_counts: dict, oc_counts: dict):
    global co_occ_edges_global
    print("\n  [Build] Co-occurrence edges (Ochiai * Conf * Log(AB)) ...")
    
    # Đổi tên biến local nhưng giữ nguyên tên biến global để không hỏng code cũ
    new_edges = defaultdict(list)

    for (a, b), cnt in tqdm(co_counts.items(), desc="  Calculating", ncols=80):
        if oc_counts.get(a, 0) == 0 or oc_counts.get(b, 0) == 0:
            continue
            
        # 1. Tính Ochiai (Cosine Similarity) - Đối xứng
        ochiai = cnt / math.sqrt(oc_counts[a] * oc_counts[b])
        
        # 2. Tính Tần suất Log (Popularity Bonus) - Đối xứng
        log_ab = math.log1p(cnt) # tương đương math.log(1 + cnt)
        
        # 3. Tính Confidence (Bất đối xứng -> Đồ thị có hướng)
        conf_a_b = cnt / oc_counts[a]  # Xác suất mua B khi đã có A
        conf_b_a = cnt / oc_counts[b]  # Xác suất mua A khi đã có B
        
        # 4. Công thức của User
        score_a_b = ochiai * conf_a_b * log_ab
        score_b_a = ochiai * conf_b_a * log_ab

        # Tính bonus Reorder
        rr_bonus = 1.0 + 0.3 * (reorder_rate.get(a, 0) + reorder_rate.get(b, 0)) / 2.0

        # Tính Cross-dept bonus
        dept_a = _get_dept(a)
        dept_b = _get_dept(b)
        if dept_a != dept_b and dept_a != -1 and dept_b != -1:
            cross_bonus = CROSS_DEPT_BONUS
        else:
            cross_bonus = 0.8

        # Áp dụng trọng số cuối cùng cho chiều A -> B
        w_a_b = score_a_b * rr_bonus * cross_bonus
        new_edges[a].append((b, w_a_b))
        
        # Áp dụng trọng số cuối cùng cho chiều B -> A
        w_b_a = score_b_a * rr_bonus * cross_bonus
        new_edges[b].append((a, w_b_a))

    for pid in new_edges:
        new_edges[pid].sort(key=lambda x: -x[1])
        co_occ_edges_global[pid] = dict(new_edges[pid][:KG_TOP_EDGE_LIMIT])

    del new_edges; gc.collect()
    print(f"  [OK] Graph nodes: {len(co_occ_edges_global):,}")
    return co_occ_edges_global


# =============================================================================
#  BUOC 3 -- Knowledge Graph (chi dung canh P->P CoOcc, KHONG co canh P->Aisle/Dept)
#
#  FIX [2]: Loai bo canh P->Aisle va P->Dept
#    Ly do: canh nay lam RWR "bi keo" tu P->Aisle->P' -> chi goi y san pham
#    cung aisle, day chinh xac la nguyen nhan KG tra ve toan "frozen food"
#
#  Graph chi gom canh P->P dua tren CoOcc (da co cross-dept bonus)
#  -> RWR se di theo canh "hay mua kem" thay vi canh "cung danh muc"
# =============================================================================
def _build_graph(co_occ_edges_dict: dict):
    global graph_nb, graph_wt
    print("\n  [Build] KG (P->P CoOcc only, no meta edges) ...")

    _nb = defaultdict(list)
    _wt = defaultdict(list)

    # Chi them canh P->P (CoOcc co cross-dept bonus)
    # KHONG them canh P->Aisle hoac P->Dept nua
    for pid, nbrs_dict in tqdm(co_occ_edges_dict.items(),
                                desc="  CoOcc P->P edges", ncols=80):
        for npid, w in nbrs_dict.items():
            _nb[pid].append(npid)
            _wt[pid].append(float(w))

    print("  Normalizing weights ...")
    for node in list(_nb.keys()):
        wt = np.array(_wt[node], dtype=np.float32)
        wt /= (wt.sum() + 1e-9)
        graph_nb[node] = np.array(_nb[node], dtype=np.int32)
        graph_wt[node] = wt

    del _nb, _wt; gc.collect()
    n_nodes = len(graph_nb)
    n_edges = sum(len(v) for v in graph_nb.values())
    print(f"  [OK] KG: {n_nodes:,} nodes | {n_edges:,} edges (P->P only)")


# =============================================================================
#  Build all
# =============================================================================
def build_all():
    _init_dept_map()
    _build_cb_vectors()
    co_counts, oc_counts = _build_stats_and_cooccurrence()
    co_occ_dict = _build_spmi_edges(co_counts, oc_counts)
    del co_counts, oc_counts; gc.collect()
    _build_graph(co_occ_dict)
    gc.collect()
    print("\n[OK] All models ready.")


# =============================================================================
#  DIVERSITY FILTER
#  Dam bao danh sach goi y co nhieu danh muc khac nhau
#  (khong phai toan "frozen" hay toan "produce")
#
#  Thuat toan: Greedy diversity selection
#    - Duyet ung vien theo score giam dan
#    - Dem so san pham da chon theo dept
#    - Neu dept nay da co >= MAX_PER_DEPT san pham -> bo qua (tru cung dept voi seed)
# =============================================================================
MAX_PER_DEPT       = 5    # Tang tu 3 len 5
SAME_DEPT_MAX      = 4    # Tang tu 2 len 4


def _diversity_filter(seed_pid: int, ranked_pids: list, k: int) -> list:
    seed_dept  = _get_dept(seed_pid)
    dept_count = defaultdict(int)
    result     = []

    for pid in ranked_pids:
        if len(result) >= k:
            break
        d = _get_dept(pid)
        if d == seed_dept:
            # Cung dept voi seed: giu nhung gioi han so luong
            if dept_count[d] < SAME_DEPT_MAX:
                dept_count[d] += 1
                result.append(pid)
        else:
            # Khac dept: giu nhung gioi han so luong
            if dept_count[d] < MAX_PER_DEPT:
                dept_count[d] += 1
                result.append(pid)

    return result


# =============================================================================
#  MODEL 1: Content-Based
#  Tim san pham co noi dung tuong tu, loc theo dept direction
#  Luu y: day la san pham SUBSTITUTE (tuong tu), khong phai complementary
#  Dung nhu 1 baseline de so sanh
# =============================================================================
def content_based_rec(product_id: int, k: int = 100) -> list:
    if product_id not in cb_vectors:
        return []
    scores = {}
    for pid in dl.frequent_items:
        if pid == product_id:
            continue
        sim = _cb_similarity(product_id, pid)
        if sim > 0:
            scores[pid] = sim
    recs = sorted(scores, key=lambda p: -scores[p])
    recs = dd.filter_by_direction(product_id, recs)
    # CB khong can diversity filter vi no tim san pham tuong tu (bai toan khac)
    return recs[:k]


# =============================================================================
#  MODEL 2: Collaborative (CoOcc voi cross-dept bonus)
#  Uu tien san pham hay mua kem (complementary) nhờ cross-dept bonus
# =============================================================================
def collab_rec(product_id: int, k: int = 100) -> list:
    if product_id not in co_occ_edges_global:
        return []
    scores = co_occ_edges_global[product_id]
    recs   = sorted(scores, key=lambda p: -scores[p])
    recs   = dd.filter_by_direction(product_id, recs)
    # Diversity filter: dam bao nhieu danh muc
    return _diversity_filter(product_id, recs, k)


# =============================================================================
#  MODEL 3: Knowledge Graph (RWR tren P->P CoOcc graph)
#  Graph chi co canh P->P nen RWR se di theo "hay mua kem"
# =============================================================================
def _multi_hop_rwr(seed: int) -> dict:
    visit = defaultdict(float)
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


def kg_rec(product_id: int, k: int = 100) -> list:
    if product_id not in graph_nb:
        return []
    scores = _multi_hop_rwr(product_id)
    recs   = sorted(
        (p for p in scores if p in dl.frequent_items and p != product_id),
        key=lambda p: -scores[p]
    )
    recs = dd.filter_by_direction(product_id, recs)
    return _diversity_filter(product_id, recs, k)


# =============================================================================
#  MODEL 4: Hybrid (CoOcc + KG + CB filter)
#
#  Pipeline:
#    1. Lay ung vien tu CF (CoOcc) va KG (RWR)
#    2. LOC LOAI san pham tuong tu qua CB: neu CB sim > nguong -> bo qua
#       (san pham tuong tu = substitute, khong phai complement)
#    3. Ket hop diem CoOcc va KG
#    4. Loc dept direction + diversity filter
# =============================================================================
def hybrid_rec(product_id: int, k: int = 100) -> list:
    if product_id not in graph_nb and product_id not in co_occ_edges_global:
        return []

    # 1. CF scores (CoOcc)
    coocc_scores = co_occ_edges_global.get(product_id, {})
    max_coocc    = max(coocc_scores.values()) if coocc_scores else 1.0

    # 2. KG scores (RWR)
    kg_scores = _multi_hop_rwr(product_id)
    max_kg    = max(kg_scores.values()) if kg_scores else 1.0

    pool = set(coocc_scores.keys()) | set(kg_scores.keys())

    hybrid_scores = {}
    for pid in pool:
        if pid == product_id or pid not in dl.frequent_items:
            continue

        # 3. CB filter: loai san pham TUONG TU (substitute)
        sim = _cb_similarity(product_id, pid)
        if sim >= CB_FILTER_HIGH:
            continue  # Qua giong = substitute -> loai

        # 4. Cross-dept bonus truc tiep trong score cuoi
        dept_a = _get_dept(product_id)
        dept_b = _get_dept(pid)
        if dept_a != dept_b and dept_a != -1 and dept_b != -1:
            dept_multiplier = 1.4   # uu tien complementary
        else:
            dept_multiplier = 0.5   # giam sat san pham cung dept

        norm_coocc = coocc_scores.get(pid, 0.0) / (max_coocc + 1e-9)
        norm_kg = kg_scores.get(pid, 0.0) / (max_kg + 1e-9)

        score = (HYB_ALPHA * norm_coocc + (1.0 - HYB_ALPHA) * norm_kg) * dept_multiplier
        hybrid_scores[pid] = score

    recs = sorted(hybrid_scores, key=lambda p: -hybrid_scores[p])
    recs = dd.filter_by_direction(product_id, recs)
    return _diversity_filter(product_id, recs, k)