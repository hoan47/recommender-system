"""
Numba-accelerated operations cho các model.
Giúp tăng tốc: (1) đếm co-occurrence pairs, (2) random walk alias sampling.

Các hàm trong file này được compile JIT bởi Numba, KHÔNG dùng Python objects
như dict, list of tuples,... — chỉ dùng numpy arrays + primitive types.
"""
import numpy as np
from numba import njit, prange


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
# Node2Vec: graph adjacency CSR-like helpers
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

    return indptr, neighbors, weights


# ===========================================================================
# Node2Vec: alias sampling
# ===========================================================================

@njit
def alias_setup(probs):
    """
    Setup alias tables cho O(1) sampling từ discrete distribution.

    Parameters
    ----------
    probs : np.ndarray (float64)
        Probabilities (không cần normalize).

    Returns
    -------
    prob_norm : np.ndarray (float64)
    alias : np.ndarray (int32)
    """
    n = len(probs)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32)

    # Normalize
    total = probs.sum()
    if total == 0:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.int32)

    norm_probs = probs / total
    prob_norm = np.zeros(n, dtype=np.float64)
    alias = np.zeros(n, dtype=np.int32)

    # Khởi tạo small và large lists
    n_small = 0
    n_large = 0
    small = np.zeros(n, dtype=np.int32)
    large = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if norm_probs[i] < 1.0 / n:
            small[n_small] = i
            n_small += 1
        else:
            large[n_large] = i
            n_large += 1

    while n_small > 0 and n_large > 0:
        n_small -= 1
        s = small[n_small]
        n_large -= 1
        l = large[n_large]

        prob_norm[s] = norm_probs[s] * n
        alias[s] = l
        norm_probs[l] = norm_probs[l] + norm_probs[s] - 1.0 / n

        if norm_probs[l] < 1.0 / n:
            small[n_small] = l
            n_small += 1
        else:
            large[n_large] = l
            n_large += 1

    while n_small > 0:
        n_small -= 1
        prob_norm[small[n_small]] = 1.0
        alias[small[n_small]] = small[n_small]

    while n_large > 0:
        n_large -= 1
        prob_norm[large[n_large]] = 1.0
        alias[large[n_large]] = large[n_large]

    return prob_norm, alias


@njit
def alias_draw(prob_norm, alias, rng_state):
    """
    Draw 1 sample từ alias tables.

    Parameters
    ----------
    prob_norm : np.ndarray (float64)
    alias : np.ndarray (int32)
    rng_state : np.ndarray (int64, shape=(2,))
        State cho numpy random (seed, ...).

    Returns
    -------
    sample : int32
    rng_state : np.ndarray (int64, shape=(2,))
        Updated state.
    """
    n = len(prob_norm)
    if n == 0:
        return -1, rng_state

    # Uniform random [0, n)
    # Dùng xorshift+ style generator đơn giản
    rng_state[0] ^= (rng_state[0] << 21)
    rng_state[0] ^= (rng_state[0] >> 35)
    rng_state[0] ^= (rng_state[0] << 4)
    rng_state[1] += 1
    r = (rng_state[0] + rng_state[1]) % 1000000007
    idx = int(r % n)  # uniform int trong [0, n)

    # Second uniform [0, 1)
    rng_state[0] ^= (rng_state[0] << 21)
    rng_state[0] ^= (rng_state[0] >> 35)
    rng_state[0] ^= (rng_state[0] << 4)
    rng_state[1] += 1
    r2 = (rng_state[0] + rng_state[1]) % 1000000007
    u = r2 / 1000000007.0

    if u < prob_norm[idx]:
        return idx, rng_state
    else:
        return alias[idx], rng_state


# ===========================================================================
# Node2Vec: random walk
# ===========================================================================

@njit
def random_walk_node2vec(indptr, neighbors, weights, p, q, walk_length,
                         start_node, rng_state):
    """
    Sinh 1 random walk từ start_node trên graph CSR.

    Parameters
    ----------
    indptr : np.ndarray (int32) (n_nodes+1,)
    neighbors : np.ndarray (int32)
    weights : np.ndarray (int32)
    p : float — return parameter
    q : float — in-out parameter
    walk_length : int
    start_node : int — node bắt đầu (index)
    rng_state : np.ndarray (int64, shape=(2,))

    Returns
    -------
    walk : np.ndarray (int32) (walk_length,) — padded với -1 nếu walk ngắn hơn
    walk_len : int — độ dài thực tế
    rng_state : np.ndarray (int64, shape=(2,))
    """
    walk = np.zeros(walk_length, dtype=np.int32)
    walk[0] = start_node
    walk_len = 1

    cur = start_node
    for step in range(1, walk_length):
        # Lấy neighbors của cur từ CSR
        start = indptr[cur]
        end = indptr[cur + 1]
        n_neighbors = end - start

        if n_neighbors == 0:
            break

        cur_neighbors = neighbors[start:end]
        cur_weights = weights[start:end]

        if step == 1:
            # Bước đầu: uniform random
            idx, rng_state = alias_draw(
                np.ones(n_neighbors, dtype=np.float64),
                np.arange(n_neighbors, dtype=np.int32),
                rng_state
            )
            if idx >= 0:
                next_node = cur_neighbors[idx]
            else:
                break
        else:
            prev = walk[step - 2]
            # Tính adjusted weights dựa trên p, q
            adj_weights = np.zeros(n_neighbors, dtype=np.float64)
            for i in range(n_neighbors):
                nb = cur_neighbors[i]
                w = float(cur_weights[i])
                if nb == prev:
                    adj_weights[i] = w / p
                elif _is_common_neighbor_numba(nb, prev, cur,
                                               indptr, neighbors):
                    adj_weights[i] = w
                else:
                    adj_weights[i] = w / q

            # Setup alias
            prob_norm, alias_tbl = alias_setup(adj_weights)
            idx, rng_state = alias_draw(prob_norm, alias_tbl, rng_state)
            if idx >= 0:
                next_node = cur_neighbors[idx]
            else:
                break

        walk[step] = next_node
        cur = next_node
        walk_len = step + 1

    return walk, walk_len, rng_state


@njit
def _is_common_neighbor_numba(node, prev, cur, indptr, neighbors):
    """
    Kiểm tra node có phải là common neighbor của prev và cur không.
    Nghĩa là node ∈ N(prev) ∩ N(cur), ngoại trừ chính node.
    """
    # Lấy neighbors của prev
    start_prev = indptr[prev]
    end_prev = indptr[prev + 1]
    # Lấy neighbors của cur
    start_cur = indptr[cur]
    end_cur = indptr[cur + 1]

    # Duyệt neighbors của cur, kiểm tra trong neighbors của prev
    # (tối ưu: luôn duyệt cái ngắn hơn — nhưng để đơn giản duyệt cur)
    for i in range(start_cur, end_cur):
        if neighbors[i] == node:
            # Node là neighbor của cur → kiểm tra có trong prev ko
            for j in range(start_prev, end_prev):
                if neighbors[j] == node:
                    return True
            return False
    return False


@njit
def generate_walks_numba(indptr, neighbors, weights, p, q, walk_length,
                         num_walks, nodes, rng_seed):
    """
    Sinh random walks cho tất cả nodes — dùng parallel.

    Parameters
    ----------
    indptr, neighbors, weights : graph CSR
    p, q : float
    walk_length : int
    num_walks : int
    nodes : np.ndarray (int32) — list các node indices cần walk
    rng_seed : int

    Returns
    -------
    walks : np.ndarray (int32) shape (n_total_walks, walk_length)
        -1 padding cho walks ngắn hơn walk_length.
    walk_lengths : np.ndarray (int32) shape (n_total_walks,)
        Độ dài thực tế của mỗi walk.
    """
    n_nodes = len(nodes)
    total_walks = n_nodes * num_walks

    walks = np.full((total_walks, walk_length), -1, dtype=np.int32)
    walk_lengths = np.zeros(total_walks, dtype=np.int32)

    for walk_idx in range(total_walks):
        node_idx = walk_idx % n_nodes
        start_node = nodes[node_idx]

        # Tạo rng_state riêng cho mỗi walk (tránh race condition)
        # Dùng walk_idx làm seed variant
        rng_state = np.array([int(rng_seed) + walk_idx * 1234567,
                              int(rng_seed) + walk_idx * 7654321 + 1],
                             dtype=np.int64)

        walk, wlen, _ = random_walk_node2vec(
            indptr, neighbors, weights, p, q, walk_length,
            start_node, rng_state
        )

        walks[walk_idx, :wlen] = walk[:wlen]
        walk_lengths[walk_idx] = wlen

    return walks, walk_lengths