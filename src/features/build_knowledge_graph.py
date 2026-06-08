"""
Knowledge Graph — Node2Vec embeddings từ đồ thị product-department

Xây dựng đồ thị với 2 loại node:
  - Product nodes (p0..pN): đại diện cho sản phẩm
  - Department nodes (d0..dM): đại diện cho ngành hàng

Edge gồm:
  1. co_purchase: product-product, weight = SPMI score (từ build_spmi)
     Chỉ giữ cặp có SPMI > 0 để lọc nhiễu
  2. belongs_to: product-department, weight = 1.0 (từ products.csv)
  3. dept_co_purchase: department-department, weight = department-level SPMI
     Cho phép cross-department recommendation (vd: gà → xà) qua đường vòng
     d_thực_phẩm → d_rau_củ → p_xà

Sau đó dùng node2vec (random walks + skip-gram negative sampling)
để học product embeddings.

Optimizations:
  1. Graph → adjacency lists (list of neighbor arrays + weight CDFs) 
     để Numba JIT có thể random walk trực tiếp
  2. Skip-gram training loop đẩy vào Numba JIT (njit, parallel=False)
  3. Noise distribution cho negative sampling precompute + JIT sampling
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import math
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from scipy.sparse import load_npz, save_npz, lil_matrix, csr_matrix
from numba import njit
from numba.typed import List as NumbaList
from tqdm import tqdm

from src.config import MODELS_DIR, DATA_DIR, KG_DIM, KG_WALK_LENGTH, KG_NUM_WALKS, KG_EPOCHS

# File lưu embeddings và similarity matrix
EMB_FILE = MODELS_DIR / "kg_embeddings.npy"
SIM_FILE = MODELS_DIR / "kg_similarity.npz"
DEPT_SPMI_FILE = MODELS_DIR / "dept_spmi.npz"


def build_dept_spmi(prior, products_df):
    """
    Tính SPMI (Shifted Positive Pointwise Mutual Information) giữa các department.
    
    Công thức SPMI (cùng level product):
      P(d_k, d_l) = số đơn hàng chứa cả d_k và d_l / tổng số đơn
      P(d_k)       = số đơn hàng chứa d_k / tổng số đơn
      PMI = log2( P(d_k, d_l) / (P(d_k) * P(d_l)) )
      SPMI = max(0, PMI - log2(alpha)), với alpha = 1 (Shifted)
    
    Tham số:
        prior: DataFrame — order_products__prior (order_id, product_id)
        products_df: DataFrame — product_id, department_id
    
    Trả về:
        dept_spmi: csr_matrix (n_depts x n_depts) — SPMI giữa các department
        n_depts: int — số department
    """
    print("\n  [KG] Computing department-department SPMI ...")
    n_depts = products_df["department_id"].nunique()
    
    # Map product_id → department_id
    p2d = dict(zip(products_df["product_id"], products_df["department_id"]))
    
    # Với mỗi order, lấy danh sách department_id duy nhất
    order_depts = prior.groupby("order_id")["product_id"].apply(
        lambda prods: list(set(p2d.get(p, -1) for p in prods if p2d.get(p, -1) != -1))
    )
    n_orders = len(order_depts)
    print(f"  [KG]   {n_orders:,} orders, {n_depts} departments")
    
    # Đếm co-occurrence giữa các department
    # Dùng ma trận thưa 21×21
    cooc = lil_matrix((n_depts, n_depts), dtype=np.float64)
    dept_count = np.zeros(n_depts, dtype=np.float64)
    
    for depts in order_depts:
        # Đánh dấu department nào xuất hiện trong order này
        depts_unique = set(depts)
        for d in depts_unique:
            dept_count[d - 1] += 1.0  # department_id bắt đầu từ 1
        # Tăng co-occurrence cho từng cặp department trong order
        dept_list = sorted(depts_unique)
        for i_idx, d_i in enumerate(dept_list):
            for d_j in dept_list[i_idx:]:  # i_idx để lấy tam giác trên (d_i <= d_j)
                cooc[d_i - 1, d_j - 1] += 1.0
    
    # Tính SPMI
    n_orders_f = float(n_orders)
    alpha = 1.0  # Shifted parameter
    spmi = lil_matrix((n_depts, n_depts), dtype=np.float32)
    
    coo = cooc.tocoo()
    for i, j, p_ij in zip(coo.row, coo.col, coo.data):
        p_i = dept_count[i] / n_orders_f
        p_j = dept_count[j] / n_orders_f
        p_ij_norm = p_ij / n_orders_f
        
        pmi = math.log2(p_ij_norm / (p_i * p_j)) if p_i > 0 and p_j > 0 else 0.0
        spmi_val = max(0.0, pmi - math.log2(alpha))
        
        if spmi_val > 0:
            spmi[i, j] = spmi_val
            if i != j:
                spmi[j, i] = spmi_val  # Đối xứng
    
    csr_spmi = spmi.tocsr()
    print(f"  [KG]   Dept SPMI: {csr_spmi.nnz:,} non-zero pairs (density {csr_spmi.nnz / (n_depts * n_depts):.2%})")
    
    # Lưu lại để dùng sau
    save_npz(DEPT_SPMI_FILE, csr_spmi)
    
    del cooc, spmi; gc.collect()
    return csr_spmi, n_depts


def build_graph(spmi, dept_spmi, products_df, n_depts):
    """
    Xây dựng đồ thị NetworkX từ SPMI matrix, department SPMI và product metadata.
    
    Tham số:
        spmi: csr_matrix — ma trận SPMI product-product từ build_spmi
        dept_spmi: csr_matrix — ma trận SPMI department-department
        products_df: DataFrame — product_id, department_id
        n_depts: int — số department
        
    Trả về:
        G: nx.Graph — đồ thị vô hướng
    """
    print("\n  [KG] Building graph ...")
    G = nx.Graph()
    n = spmi.shape[0]
    
    # Thêm product nodes (p0..pn-1) với type="product"
    G.add_nodes_from([f"p{i}" for i in range(n)], type="product")
    
    # Thêm department nodes (d0..dM-1) với type="department"
    G.add_nodes_from([f"d{i}" for i in range(n_depts)], type="department")
    
    # Thêm edges co_purchase từ SPMI matrix (chỉ tam giác trên i<j)
    coo = spmi.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i < j and w > 0:
            G.add_edge(f"p{i}", f"p{j}", weight=float(w))
    
    # Thêm edges belongs_to: product → department
    for _, row in products_df.iterrows():
        pid, did = int(row["product_id"]), int(row["department_id"])
        if pid < n:
            G.add_edge(f"p{pid}", f"d{did - 1}", weight=1.0)  # did-1 vì department SPMI dùng 0-based
    
    # Thêm edges department-department từ dept_spmi
    coo_d = dept_spmi.tocoo()
    for i, j, w in zip(coo_d.row, coo_d.col, coo_d.data):
        if i <= j and w > 0:  # Tam giác trên để tránh duplicate
            G.add_edge(f"d{i}", f"d{j}", weight=float(w))
    
    print(f"  [KG] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def _graph_to_adjacency(G):
    """
    Chuyển NetworkX graph thành adjacency arrays cho Numba JIT.
    
    Mỗi node được biểu diễn bởi:
      - neighbor_ids: int32 array of neighbors
      - weight_cdf: float64 array of cumulative weight distribution
    
    Trả về:
        node_ids: list[str] — danh sách node labels theo index
        n2i: dict — label → index
        neigh_list: NumbaList[int32[:]] — neighbors cho mỗi node
        cdf_list: NumbaList[float64[:]] — cumulative weight distribution
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    n2i = {n: i for i, n in enumerate(nodes)}
    
    neigh_list = NumbaList()
    cdf_list = NumbaList()
    
    for node in nodes:
        nb = list(G.neighbors(node))
        if not nb:
            neigh_list.append(np.empty(0, dtype=np.int32))
            cdf_list.append(np.empty(0, dtype=np.float64))
            continue
        
        # Chuyển neighbor labels → indices
        nb_indices = np.array([n2i[n] for n in nb], dtype=np.int32)
        weights = np.array([G[node][n].get("weight", 1.0) for n in nb], dtype=np.float64)
        
        # Normalize weights → CDF
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        cdf = np.cumsum(weights)
        
        neigh_list.append(nb_indices)
        cdf_list.append(cdf)
    
    return nodes, n2i, neigh_list, cdf_list


@njit
def _random_walks_jit(n_nodes, n_walks, walk_len, neigh_list, cdf_list, rng_state):
    """
    Sinh random walks từ graph adjacency dùng Numba JIT.
    Bước tiếp theo được chọn dựa trên edge weight (weighted sampling).
    
    Tham số:
        n_nodes: int
        n_walks: int — số walk mỗi node
        walk_len: int — độ dài mỗi walk
        neigh_list: List[int32[:]] — neighbors mỗi node
        cdf_list: List[float64[:]] — cumulative distribution
        rng_state: mảng trạng thái RNG (numba random)
    
    Trả về:
        walks_flat: int32[:] — tất cả walks được flatten
        walk_lengths: int32[:] — độ dài thực tế của mỗi walk
    """
    np.random.seed(rng_state[0])
    
    total_steps = n_nodes * n_walks * walk_len
    walks_flat = np.empty(total_steps, dtype=np.int32)
    walk_lengths = np.empty(n_nodes * n_walks, dtype=np.int32)
    
    out_idx = 0
    wlk_idx = 0
    
    for start_node in range(n_nodes):
        for _ in range(n_walks):
            cur = start_node
            walk_start = out_idx
            walks_flat[out_idx] = cur
            out_idx += 1
            
            for step in range(1, walk_len):
                nbs = neigh_list[cur]
                cdf = cdf_list[cur]
                if len(nbs) == 0:
                    break
                
                # Weighted sampling: uniform random → binary search on CDF
                r = np.random.random()
                # Linear scan (nhanh cho degree nhỏ, vốn có trong graph này)
                chosen = nbs[-1]  # default to last
                for k in range(len(cdf)):
                    if r <= cdf[k]:
                        chosen = nbs[k]
                        break
                
                cur = chosen
                walks_flat[out_idx] = cur
                out_idx += 1
            
            walk_lengths[wlk_idx] = out_idx - walk_start
            wlk_idx += 1
    
    # Trim excess
    walks_flat = walks_flat[:out_idx]
    return walks_flat, walk_lengths


@njit
def _skipgram_train(walks_flat, walk_lengths, walk_offsets, n_nodes, dim, 
                     epochs, lr_init, noise, neg_samples):
    """
    Skip-gram training với negative sampling dùng Numba JIT.
    
    Tham số:
        walks_flat: int32[:] — flattened walk sequences
        walk_lengths: int32[:] — độ dài mỗi walk
        walk_offsets: int32[:] — start index của mỗi walk trong walks_flat
        n_nodes: int — tổng số nodes
        dim: int — kích thước embedding
        epochs: int — số epoch
        lr_init: float — learning rate ban đầu
        noise: float64[:] — noise distribution
        neg_samples: int — số negative samples mỗi positive pair
    
    Trả về:
        W_in: float32[:,:] — center embeddings
    """
    n_walks = len(walk_lengths)
    window = dim // 10  # Context window size
    
    # Xavier init
    W_in = np.random.uniform(-0.5 / dim, 0.5 / dim, (n_nodes, dim)).astype(np.float32)
    W_out = np.random.uniform(-0.5 / dim, 0.5 / dim, (n_nodes, dim)).astype(np.float32)
    
    for epoch in range(epochs):
        lr = lr_init * (0.95 ** epoch)
        total_loss = 0.0
        
        for w_idx in range(n_walks):
            start = walk_offsets[w_idx]
            end = start + walk_lengths[w_idx]
            
            for pos in range(start, end):
                center = walks_flat[pos]
                
                # Context window
                left = max(start, pos - window)
                right = min(end - 1, pos + window)
                
                for ctx_pos in range(left, right + 1):
                    if ctx_pos == pos:
                        continue
                    ctx = walks_flat[ctx_pos]
                    
                    # Positive: (center, context)
                    dot = 0.0
                    for k in range(dim):
                        dot += W_in[center, k] * W_out[ctx, k]
                    sig = 1.0 / (1.0 + math.exp(-dot))
                    grad = sig - 1.0
                    total_loss += -math.log(max(sig, 1e-15))
                    
                    # Update positive
                    for k in range(dim):
                        grad_w_in = lr * grad * W_out[ctx, k]
                        grad_w_out = lr * grad * W_in[center, k]
                        W_in[center, k] -= grad_w_in
                        W_out[ctx, k] -= grad_w_out
                    
                    # Negative samples
                    for _ in range(neg_samples):
                        neg = np.searchsorted(noise, np.random.random())
                        if neg == center or neg == ctx:
                            continue
                        
                        dot_n = 0.0
                        for k in range(dim):
                            dot_n += W_in[center, k] * W_out[neg, k]
                        sig_n = 1.0 / (1.0 + math.exp(-dot_n))
                        grad_n = sig_n
                        total_loss += -math.log(max(1.0 - sig_n, 1e-15))
                        
                        # Update negative
                        for k in range(dim):
                            grad_w_in = lr * grad_n * W_out[neg, k]
                            grad_w_out = lr * grad_n * W_in[center, k]
                            W_in[center, k] -= grad_w_in
                            W_out[neg, k] -= grad_w_out
        
        # Làm tròn thủ công vì Numba không hỗ trợ string formatting
        loss_str = int(total_loss * 10000) / 10000
        print("    Epoch", epoch + 1, "/", epochs, "loss=", loss_str)
    
    return W_in


def train_node2vec(G, n_products, dim=KG_DIM, walk_len=KG_WALK_LENGTH, 
                   n_walks=KG_NUM_WALKS, epochs=KG_EPOCHS, neg_samples=5):
    """
    Huấn luyện node2vec embeddings với Numba JIT acceleration.
    
    Bước 1: Chuyển graph → adjacency arrays
    Bước 2: Random walks bằng Numba JIT
    Bước 3: Skip-gram training bằng Numba JIT
    
    Tham số:
        G: nx.Graph — đồ thị đầu vào
        n_products: int — số lượng product
        dim: int — kích thước embedding
        walk_len: int — độ dài mỗi walk
        n_walks: int — số walk mỗi node
        epochs: int — số epoch
        neg_samples: int — số negative samples
    
    Trả về:
        emb: numpy array (n_products x dim) — product embeddings
    """
    print(f"  [KG] Training node2vec dim={dim} (Numba JIT) ...")
    
    # Bước 1: Chuyển graph sang adjacency arrays
    print("  [KG]   Converting graph to adjacency lists ...")
    nodes, n2i, neigh_list, cdf_list = _graph_to_adjacency(G)
    n_nodes = len(nodes)
    
    # Bước 2: Random walks
    print(f"  [KG]   Generating random walks ({n_nodes:,} nodes × {n_walks} walks × {walk_len} steps) ...")
    rng_state = np.array([np.random.randint(0, 2**31)], dtype=np.int64)
    walks_flat, walk_lengths = _random_walks_jit(
        n_nodes, n_walks, walk_len, neigh_list, cdf_list, rng_state
    )
    
    # Precompute walk offsets
    n_total_walks = len(walk_lengths)
    walk_offsets = np.empty(n_total_walks + 1, dtype=np.int32)
    walk_offsets[0] = 0
    for i in range(n_total_walks):
        walk_offsets[i + 1] = walk_offsets[i] + walk_lengths[i]
    
    print(f"  [KG]   {n_total_walks:,} walks, total steps: {walk_offsets[-1]:,}")
    
    # Bước 3: Compute noise distribution
    freq = np.zeros(n_nodes, dtype=np.float64)
    for w_idx in range(n_total_walks):
        s = walk_offsets[w_idx]
        e = walk_offsets[w_idx + 1]
        for p in range(s, e):
            freq[walks_flat[p]] += 1.0
    noise = freq ** 0.75
    noise_sum = noise.sum()
    if noise_sum > 0:
        noise /= noise_sum
    # Convert noise to CDF for binary search in JIT
    noise_cdf = np.cumsum(noise)
    
    # Bước 4: Skip-gram training (Numba JIT)
    print(f"  [KG]   Training skip-gram ({epochs} epoch(s)) ...")
    W_in = _skipgram_train(
        walks_flat, walk_lengths, walk_offsets, n_nodes, dim,
        epochs, 0.025, noise_cdf, neg_samples
    )
    
    # Bước 5: Trích xuất product embeddings
    emb = np.zeros((n_products, dim), dtype=np.float32)
    for pid in range(n_products):
        label = f"p{pid}"
        if label in n2i:
            emb[pid] = W_in[n2i[label]].astype(np.float32)
    
    del walks_flat, walk_lengths, walk_offsets, W_in
    gc.collect()
    return emb


def compute_similarity(emb, top_k=100, chunk=1000):
    """
    Tính cosine similarity giữa các product embeddings.
    Dùng chunked computation để tránh full dense matrix (50K×50K = 10GB).
    
    Tham số:
        emb: numpy array (n_products x dim) — product embeddings
        top_k: int — chỉ giữ top-K tương tự mỗi dòng
        chunk: int — kích thước chunk để tính (mặc định 1000)
        
    Trả về:
        csr_matrix (n_products x n_products) — similarity sparse matrix
    """
    n = emb.shape[0]
    # L2 normalize embeddings (cần thiết cho cosine similarity)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Tránh chia 0 cho cold-start
    emb_norm = emb / norms
    
    sim = lil_matrix((n, n), dtype=np.float32)
    for start in tqdm(range(0, n, chunk), desc="  Similarity"):
        end = min(start + chunk, n)
        chunk_sim = emb_norm[start:end] @ emb_norm.T  # Dot product
        for i_local, row_idx in enumerate(range(start, end)):
            row = chunk_sim[i_local]
            row[row_idx] = 0  # Bỏ self-similarity
            if top_k < n:
                # Chỉ giữ top-K giá trị dương
                idx = np.argpartition(row, -top_k)[-top_k:]
                vals = row[idx]
                mask = vals > 0
                if mask.any():
                    sim[row_idx, idx[mask]] = vals[mask].astype(np.float32)
            else:
                mask = row > 0
                if mask.any():
                    sim[row_idx, mask] = row[mask].astype(np.float32)
    csr = sim.tocsr()
    del sim; gc.collect()
    return csr


def save(emb, sim):
    """Lưu embeddings và similarity matrix"""
    np.save(EMB_FILE, emb)
    save_npz(SIM_FILE, sim)
    print(f"  [KG] Saved: {EMB_FILE}, {SIM_FILE}")


def load():
    """Tải embeddings và similarity matrix"""
    return np.load(EMB_FILE), load_npz(SIM_FILE)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(MODELS_DIR.parent))
    from src.data_loader import load_products, load_prior
    products = load_products()
    prior = load_prior()
    # Tải SPMI matrix từ bước trước
    spmi = load_npz(MODELS_DIR / "spmi_matrix.npz")
    # Tính department-department SPMI
    dept_spmi, n_depts = build_dept_spmi(prior, products)
    # Xây dựng đồ thị với department-department edges
    G = build_graph(spmi, dept_spmi, products, n_depts)
    emb = train_node2vec(G, spmi.shape[0])
    sim = compute_similarity(emb)
    save(emb, sim)
