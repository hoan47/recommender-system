"""
Recommend — gợi ý sản phẩm dùng Hybrid (SPMI + KG - CB filter)
"""
import numpy as np
from scipy.sparse import load_npz

from src.config import MODELS_DIR

def recommend(pid, top_k=10):
    """
    Gợi ý top-K sản phẩm cho product_id.
    Dùng Hybrid matrix (hoặc fallback về SPMI nếu Hybrid không có).
    """
    # Thử load Hybrid trước
    try:
        matrix = load_npz(MODELS_DIR / "hybrid_matrix.npz")
    except:
        matrix = load_npz(MODELS_DIR / "spmi_matrix.npz")

    row = matrix[pid]
    if row.nnz == 0:
        return []

    order = np.argsort(row.data)[::-1]
    top = row.indices[order[:top_k]].tolist()
    return top

def recommend_with_scores(pid, top_k=10):
    """Gợi ý kèm điểm số"""
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
    # Demo
    print("Demo recommend cho product_id=1:")
    recs = recommend_with_scores(1, 10)
    for pid, score in recs:
        print(f"  -> {pid}: {score:.4f}")