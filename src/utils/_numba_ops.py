"""
Numba-accelerated operations cho các model.
Giúp tăng tốc: (1) đếm co-occurrence pairs, (2) xây adjacency CSR.

Các hàm trong file này được compile JIT bởi Numba, KHÔNG dùng Python objects
như dict, list of tuples,... — chỉ dùng numpy arrays + primitive types.
"""
import numpy as np
from numba import njit


# ===========================================================================
# Co-occurrence pair counting
# ===========================================================================

@njit
def count_pairs_numba(order_indices, order_ptr, n_products):
    """
    Đếm co-occurrence pairs từ orders slice.

    Parameters
    ----------
    order_indices : np.ndarray (int32)
        Flat array product indices cho tất cả orders, theo thứ tự.
    order_ptr : np.ndarray (int32)
        (n_orders+1,) — start index của mỗi order trong order_indices.
        order_ptr[-1] = len(order_indices).
    n_products : int
        Tổng số sản phẩm, dùng để tính upper bound pairs.

    Returns
    -------
    rows : np.ndarray (int32)
        (n_pairs,) — row indices của co-occurrence pairs.
    cols : np.ndarray (int32)
        (n_pairs,) — column indices của co-occurrence pairs.
    counts : np.ndarray (int32)
        (n_pairs,) — số lần mỗi pair xuất hiện (có thể có duplicates).
    """
    # Pre-allocate với upper bound (worst case)
    # Mỗi order có tối đa n_items*(n_items-1)//2 pairs trong order hiện tại
    # Nhưng không biết trước → dùng 2 phase count
    n_orders = order_ptr.shape[0] - 1

    # Phase 1: đếm số pairs tổng cộng
    total_pairs = 0
    for o in range(n_orders):
        start = order_ptr[o]
        end = order_ptr[o + 1]
        n = end - start
        if n >= 2:
            total_pairs += n * (n - 1) // 2

    if total_pairs == 0:
        return (np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))

    # Phase 2: ghi pairs vào arrays
    rows = np.zeros(total_pairs, dtype=np.int32)
    cols = np.zeros(total_pairs, dtype=np.int32)

    pos = 0
    for o in range(n_orders):
        start = order_ptr[o]
        end = order_ptr[o + 1]
        n = end - start
        if n < 2:
            continue
        for i in range(start, end):
            for j in range(i + 1, end):
                a = order_indices[i]
                b = order_indices[j]
                if a != b:
                    rows[pos] = a
                    cols[pos] = b
                    pos += 1

    # Cắt bớt (nếu có a==b bị skip)
    if pos < total_pairs:
        rows = rows[:pos]
        cols = cols[:pos]

    # Reduce duplicates bằng cách sort và count
    # Dùng keys = rows * n_products + cols để sort theo row + col
    keys = rows.astype(np.int64) * n_products + cols.astype(np.int64)
    idx = np.argsort(keys)
    rows_sorted = rows[idx]
    cols_sorted = cols[idx]

    # Đếm unique
    n_unique = 1
    for i in range(1, len(rows_sorted)):
        if rows_sorted[i] != rows_sorted[i-1] or cols_sorted[i] != cols_sorted[i-1]:
            n_unique += 1

    unique_rows = np.zeros(n_unique, dtype=np.int32)
    unique_cols = np.zeros(n_unique, dtype=np.int32)
    cnt = np.zeros(n_unique, dtype=np.int32)

    unique_rows[0] = rows_sorted[0]
    unique_cols[0] = cols_sorted[0]
    cnt[0] = 1
    pos_u = 0
    for i in range(1, len(rows_sorted)):
        if rows_sorted[i] == rows_sorted[i-1] and cols_sorted[i] == cols_sorted[i-1]:
            cnt[pos_u] += 1
        else:
            pos_u += 1
            unique_rows[pos_u] = rows_sorted[i]
            unique_cols[pos_u] = cols_sorted[i]
            cnt[pos_u] = 1

    return unique_rows, unique_cols, cnt


# ===========================================================================
# Graph adjacency CSR builder (dùng chung cho DeepWalk)
# ===========================================================================

@njit
def _build_adjacency_csr(pair_rows, pair_cols, pair_counts, n_products):
    """
    Xây CSR-like adjacency từ edge list.
    Output có thể dùng cho random walk.

    Parameters
    ----------
    pair_rows, pair_cols : np.ndarray (int32)
        Edge list (undirected).
    pair_counts : np.ndarray (int32)
        Weight (co-occurrence count) cho mỗi edge.
    n_products : int

    Returns
    -------
    indptr : np.ndarray (int32) (n_products+1,)
    neighbors : np.ndarray (int32)
    weights : np.ndarray (int32)
    """
    # Đếm số neighbors cho mỗi node (cả 2 hướng)
    degree = np.zeros(n_products, dtype=np.int32)
    for i in range(len(pair_rows)):
        degree[pair_rows[i]] += 1
        degree[pair_cols[i]] += 1

    # Build indptr
    indptr = np.zeros(n_products + 1, dtype=np.int32)
    for i in range(n_products):
        indptr[i + 1] = indptr[i] + degree[i]

    # Fill neighbors và weights
    neighbors = np.zeros(indptr[-1], dtype=np.int32)
    weights = np.zeros(indptr[-1], dtype=np.int32)

    # Temp position counter cho mỗi node
    pos = np.zeros(n_products, dtype=np.int32)
    for i in range(len(pair_rows)):
        u = pair_rows[i]
        v = pair_cols[i]
        w = pair_counts[i]

        idx_u = indptr[u] + pos[u]
        neighbors[idx_u] = v
        weights[idx_u] = w
        pos[u] += 1

        idx_v = indptr[v] + pos[v]
        neighbors[idx_v] = u
        weights[idx_v] = w
        pos[v] += 1

    # Sort neighbors + weights mỗi node để dùng binary search
    for node in range(n_products):
        start = indptr[node]
        end = indptr[node + 1]
        n = end - start
        if n <= 1:
            continue
        # Sort neighbors và weights đồng thời
        idx_sorted = np.argsort(neighbors[start:end])
        neighbors[start:end] = neighbors[start:end][idx_sorted]
        weights[start:end] = weights[start:end][idx_sorted]

    return indptr, neighbors, weights