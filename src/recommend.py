"""
Recommend — gợi ý sản phẩm dùng Hybrid matrix (có fallback Confidence)
Tích hợp diversity filter + dept direction filter

Cách dùng:
    from src.recommend import recommend
    recs = recommend(product_id=1, top_k=10)  # Trả về list product_id
    recs = recommend_with_scores(1, 10)       # Trả về list (pid, score)

Giải thích luồng:
    1. Thử load Hybrid matrix (đã kết hợp Confidence + KG + CB filter)
    2. Nếu Hybrid không có → fallback về Confidence
    3. Lấy dòng product_id từ matrix, sắp xếp theo score giảm dần
    4. Filter: Department direction → Diversity → top-K
    5. Trả về K kết quả
"""

import numpy as np
from scipy.sparse import load_npz

from src.config import MODELS_DIR
from src.features.diversity import diversity_filter
from src.features.dept_direction import filter_by_direction, dept_suggest

# Cache cho matrix và prod_dept_map
_hybrid_matrix = None
_confidence_matrix = None
_prod_dept_map = None
_matrix_loaded = False


def _load_matrices():
    """Load matrices một lần, cache lại."""
    global _hybrid_matrix, _confidence_matrix, _matrix_loaded
    if _matrix_loaded:
        return
    
    try:
        _hybrid_matrix = load_npz(MODELS_DIR / "hybrid_matrix.npz")
    except FileNotFoundError:
        try:
            _confidence_matrix = load_npz(MODELS_DIR / "confidence_matrix.npz")
        except FileNotFoundError:
            pass
    
    _matrix_loaded = True


def set_prod_dept_map(dept_map):
    """Set product -> department mapping từ data_loader."""
    global _prod_dept_map
    _prod_dept_map = dept_map


def recommend(pid, top_k=10):
    """
    Gợi ý top-K sản phẩm cho product_id (tích hợp diversity + dept direction).
    
    Tham số:
        pid: int — product_id cần gợi ý
        top_k: int — số lượng gợi ý
    
    Trả về:
        list[int] — danh sách product_id gợi ý
    """
    _load_matrices()
    
    # Chọn matrix
    if _hybrid_matrix is not None:
        matrix = _hybrid_matrix
    elif _confidence_matrix is not None:
        matrix = _confidence_matrix
    else:
        print("  [Recommend] ERROR: No matrix found!")
        return []
    
    if pid >= matrix.shape[0]:
        return []
    
    row = matrix[pid]
    if row.nnz == 0:
        return []
    
    # Sắp xếp theo score giảm dần
    order = np.argsort(row.data)[::-1]
    candidates = row.indices[order].tolist()
    scores = row.data[order].tolist()
    
    # Bước 1: Department direction filter
    if _prod_dept_map is not None and dept_suggest is not None:
        candidates = filter_by_direction(pid, candidates, _prod_dept_map)
    
    # Bước 2: Diversity filter
    if _prod_dept_map is not None:
        candidates = diversity_filter(pid, candidates, _prod_dept_map, k=top_k)
    
    return candidates[:top_k]


def recommend_with_scores(pid, top_k=10):
    """
    Gợi ý top-K sản phẩm kèm điểm số (tích hợp diversity + dept direction).
    
    Tham số:
        pid: int — product_id cần gợi ý
        top_k: int — số lượng gợi ý
    
    Trả về:
        list[tuple(int, float)] — danh sách (product_id, score)
    """
    _load_matrices()
    
    if _hybrid_matrix is not None:
        matrix = _hybrid_matrix
    elif _confidence_matrix is not None:
        matrix = _confidence_matrix
    else:
        return []
    
    if pid >= matrix.shape[0]:
        return []
    
    row = matrix[pid]
    if row.nnz == 0:
        return []
    
    # Sắp xếp theo score giảm dần
    order = np.argsort(row.data)[::-1]
    candidates = row.indices[order].tolist()
    scores = row.data[order].tolist()
    
    # Bước 1: Department direction filter
    if _prod_dept_map is not None and dept_suggest is not None:
        # Lọc candidates, giữ scores tương ứng
        filtered = []
        filtered_scores = []
        for c, s in zip(candidates, scores):
            if c in filter_by_direction(pid, [c], _prod_dept_map):
                filtered.append(c)
                filtered_scores.append(s)
        candidates = filtered
        scores = filtered_scores
    
    # Bước 2: Diversity filter
    if _prod_dept_map is not None:
        candidate_depts = {d: _prod_dept_map.get(d, -1) for d in candidates}
        filtered = diversity_filter(pid, candidates, _prod_dept_map, k=top_k)
        # Chỉ lấy scores cho kết quả đã lọc
        filtered_scores = [scores[candidates.index(p)] for p in filtered]
        candidates = filtered
        scores = filtered_scores
    
    return list(zip(candidates[:top_k], scores[:top_k]))


def recommend_simple(pid, top_k=10):
    """
    Gợi ý top-K sản phẩm KHÔNG filter (dùng cho evaluation baseline).
    
    Tham số:
        pid: int — product_id cần gợi ý
        top_k: int — số lượng gợi ý
    
    Trả về:
        list[int] — danh sách product_id gợi ý
    """
    _load_matrices()
    
    if _hybrid_matrix is not None:
        matrix = _hybrid_matrix
    elif _confidence_matrix is not None:
        matrix = _confidence_matrix
    else:
        return []
    
    if pid >= matrix.shape[0]:
        return []
    
    row = matrix[pid]
    if row.nnz == 0:
        return []
    
    order = np.argsort(row.data)[::-1]
    top = row.indices[order[:top_k]].tolist()
    return top


if __name__ == "__main__":
    # Demo: gợi ý 10 sản phẩm cho product_id=1
    print("Demo recommend cho product_id=1:")
    recs = recommend_with_scores(1, 10)
    for pid, score in recs:
        print(f"  -> {pid}: {score:.4f}")