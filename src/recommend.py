"""
Recommend — gợi ý sản phẩm dùng Hybrid matrix (có fallback SPMI)

Cách dùng:
    from src.recommend import recommend
    recs = recommend(product_id=1, top_k=10)  # Trả về list product_id
    recs = recommend_with_scores(1, 10)       # Trả về list (pid, score)

Giải thích luồng:
    1. Thử load Hybrid matrix (đã kết hợp SPMI + KG + CB filter)
    2. Nếu Hybrid không có → fallback về SPMI
    3. Lấy dòng product_id từ matrix, sắp xếp theo score giảm dần
    4. Trả về top-K kết quả
"""

import numpy as np
from scipy.sparse import load_npz

from src.config import MODELS_DIR

def recommend(pid, top_k=10):
    """
    Gợi ý top-K sản phẩm cho product_id (chỉ trả về ID).
    
    Tham số:
        pid: int — product_id cần gợi ý
        top_k: int — số lượng gợi ý
    
    Trả về:
        list[int] — danh sách product_id gợi ý
    """
    # Ưu tiên Hybrid matrix, fallback về SPMI
    try:
        matrix = load_npz(MODELS_DIR / "hybrid_matrix.npz")
    except:
        matrix = load_npz(MODELS_DIR / "spmi_matrix.npz")

    row = matrix[pid]
    if row.nnz == 0:
        return []  # Cold-start: không có gợi ý

    # Sắp xếp theo score giảm dần và lấy top-K
    order = np.argsort(row.data)[::-1]
    top = row.indices[order[:top_k]].tolist()
    return top

def recommend_with_scores(pid, top_k=10):
    """
    Gợi ý top-K sản phẩm kèm điểm số.
    
    Tham số:
        pid: int — product_id cần gợi ý
        top_k: int — số lượng gợi ý
    
    Trả về:
        list[tuple(int, float)] — danh sách (product_id, score)
    """
    try:
        matrix = load_npz(MODELS_DIR / "hybrid_matrix.npz")
    except:
        matrix = load_npz(MODELS_DIR / "spmi_matrix.npz")

    row = matrix[pid]
    if row.nnz == 0:
        return []

    order = np.argsort(row.data)[::-1]
    top = [(int(row.indices[i]), float(row.data[i])) for i in order[:top_k]]
    return top

if __name__ == "__main__":
    # Demo: gợi ý 10 sản phẩm cho product_id=1
    print("Demo recommend cho product_id=1:")
    recs = recommend_with_scores(1, 10)
    for pid, score in recs:
        print(f"  -> {pid}: {score:.4f}")