"""
Hybrid — kết hợp Confidence + Knowledge Graph, loại substitute bằng CB

Công thức:
  Với mỗi cặp (seed → candidate):
    1. Lấy score từ Confidence dict (co_occ_edges) và KG dict (kg_scores)
    2. Normalize mỗi score về [0, 1] bằng cách chia cho max score của seed
    3. combined = alpha * conf_norm + (1-alpha) * kg_norm
    4. Nếu CB similarity(seed, candidate) > threshold → combined = 0 (loại substitute)
    5. Lưu vào hybrid_scores[seed] = {pid: combined}

Kĩ thuật:
  - Toàn bộ dùng dict, không sparse matrix
  - CB filter gọi inline cb_similarity() — không cần build ma trận similarity
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import json
from collections import defaultdict
from tqdm import tqdm

from src.config import MODELS_DIR
from src.config import HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH

# File lưu hybrid_scores dạng json
HYBRID_FILE = MODELS_DIR / "hybrid_dict.json"

# Biến toàn cục: {pid: {npid: combined_score}}
hybrid_scores = {}


def build_hybrid(confidence_dict, kg_dict, cb_vectors,
                 alpha=HYBRID_ALPHA, beta=HYBRID_BETA, cb_thresh=HYBRID_CB_THRESH):
    """
    Kết hợp Confidence và KG thành hybrid score, loại substitute bằng CB filter.

    Quy trình:
      1. Lấy tất cả seed products = keys(confidence) | keys(kg)
      2. Với mỗi seed:
         a. Pool = keys(confidence[seed]) | keys(kg[seed])
         b. Tính max_conf, max_kg để normalize
         c. Với mỗi candidate trong pool:
            - confidence_norm = confidence[seed][cand] / max_conf
            - kg_norm = kg[seed][cand] / max_kg (nếu có)
            - combined = alpha * conf_norm + beta * kg_norm
            - CB filter: nếu cb_similarity(seed, cand) > cb_thresh → skip
         d. Lưu kết quả
    """
    global hybrid_scores
    print(f"\n  [Hybrid] α={alpha}, β={beta}, cb_thresh={cb_thresh} ...")

    # Lấy hàm cb_similarity từ module build_cb
    from src.features.build_cb import cb_similarity

    # Tập hợp tất cả seed products
    all_seeds = set(confidence_dict.keys()) | set(kg_dict.keys())
    hybrid_scores = {}

    for seed in tqdm(all_seeds, desc="  Hybrid", ncols=80):
        # Lấy candidate pool từ cả 2 nguồn
        conf_edges = confidence_dict.get(seed, {})
        kg_edges = kg_dict.get(seed, {})

        pool = set(conf_edges.keys()) | set(kg_edges.keys())
        if not pool:
            continue

        # Tính max để normalize
        max_conf = max(conf_edges.values()) if conf_edges else 1.0
        max_kg = max(kg_edges.values()) if kg_edges else 1.0

        scores = {}
        for cand in pool:
            # Normalize scores
            conf_norm = conf_edges.get(cand, 0.0) / max_conf
            kg_norm = kg_edges.get(cand, 0.0) / max_kg

            combined = alpha * conf_norm + beta * kg_norm
            if combined <= 0:
                continue

            # CB filter: loại sản phẩm quá giống (substitute)
            if cb_thresh < 1.0:
                cb_sim = cb_similarity(seed, cand)
                if cb_sim > cb_thresh:
                    continue

            scores[cand] = round(combined, 6)

        if scores:
            hybrid_scores[seed] = scores

    del confidence_dict, kg_dict
    gc.collect()
    print(f"  [Hybrid] Done: {len(hybrid_scores):,} products with scores")


def save():
    """Lưu hybrid_scores ra file json"""
    out = {}
    for pid, edges in hybrid_scores.items():
        out[str(pid)] = {str(npid): s for npid, s in edges.items()}
    with open(HYBRID_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  [Hybrid] Saved: {HYBRID_FILE}")


def load():
    """Load hybrid_scores từ file json, chuyển key từ str về int"""
    global hybrid_scores
    with open(HYBRID_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    hybrid_scores = {
        int(pid): {int(npid): float(s) for npid, s in edges.items()}
        for pid, edges in raw.items()
    }
    print(f"  [Hybrid] Loaded: {len(hybrid_scores):,} products")


# ===== Hàm tiện ích: recommend từ hybrid_scores =====
def recommend(seed_pid, top_k=100):
    """Trả về danh sách product_id được gợi ý cho seed, giảm dần theo score"""
    edges = hybrid_scores.get(seed_pid)
    if not edges:
        return []
    sorted_pids = sorted(edges, key=lambda p: -edges[p])
    return sorted_pids[:top_k]


if __name__ == "__main__":
    from src.features.build_cb import load as load_cb

    print("  [Hybrid] Loading Confidence dict ...")
    from src.features.build_association_rules import load as load_conf, co_occ_edges
    load_conf()

    print("  [Hybrid] Loading KG scores ...")
    # KG giờ lưu dạng dict (cần kiểm tra file nào)
    # Tạm thời dùng dict rỗng nếu chưa có
    try:
        with open(MODELS_DIR / "kg_dict.json", "r") as f:
            kg_scores_raw = json.load(f)
        kg_scores = {
            int(pid): {int(npid): float(s) for npid, s in edges.items()}
            for pid, edges in kg_scores_raw.items()
        }
    except FileNotFoundError:
        print("  [Hybrid] WARNING: kg_dict.json not found, using empty KG")
        kg_scores = {}

    print("  [Hybrid] Loading CB vectors ...")
    load_cb()

    build_hybrid(co_occ_edges, kg_scores, {})
    save()