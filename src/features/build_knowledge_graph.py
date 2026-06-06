"""
Knowledge Graph (KG) model — Node2Vec embeddings.

Pipeline: random walks (node2vec strategy) + skip-gram
với negative sampling + cosine similarity. Dùng numpy + networkx + scipy.sparse.

Xây dựng đồ thị với:
  - Nodes: product (49,688) + department (21)
  - Edges:
     1. (product_A) --[co_purchase]--> (product_B): chỉ giữ cặp có SPMI > 0,
        weight = SPMI value. Dùng SPMI để lọc nhiễu và giảm edges.
     2. (product) --[belongs_to]--> (department): weight = 1.0

Học node2vec embeddings trên đồ thị, sau đó tính cosine similarity
giữa các product embeddings.

Phụ thuộc:
  - src.utils.data_loader (cho prior, products)
  - src.features.build_spmi (output: models/spmi_matrix.npz)

Outputs:
  - models/kg_embeddings.npy        - Product embeddings matrix
  - models/kg_best_params.json      - Hyperparameters tốt nhất từ grid search
  - models/kg_similarity.npz        - Cosine similarity (product x product)
"""

import json

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from tqdm import tqdm

import networkx as nx

from src.config import (
    MODELS_DIR,
    KG_WALK_LENGTHS,
    KG_DIMENSIONS_LIST,
    KG_NUM_WALKS_LIST,
    KG_WINDOW,
    KG_P,
    KG_Q,
    KG_EPOCHS,
    KG_NEGATIVE,
    KG_LR,
    KG_TOP_K,
    KG_CHUNK_SIZE,
    KG_EVAL_KS,
    KG_EMBEDDINGS_FILE,
    KG_BEST_PARAMS_FILE,
    KG_SIMILARITY_FILE,
    KG_TUNING_RESULTS_FILE,
)


# ============================================================================
# PHASE 1: Xây dựng đồ thị
# ============================================================================

def build_graph(spmi_matrix, products_df, prior_df):
    """
    Xây dựng đồ thị NetworkX với các node product và department.

    Cấu trúc đồ thị:
      - Nodes: product_<id> (type='product'), dept_<id> (type='department')
      - Edges:
        * co_purchase: product-product, weight=SPMI value (chỉ SPMI > 0)
        * belongs_to: product-department, weight=1.0

    Tham số
    ----------
    spmi_matrix : csr_matrix
        SPMI matrix (n_products x n_products).
    products_df : pd.DataFrame
        Với các cột: product_id, department_id.
    prior_df : pd.DataFrame
        Prior interactions (dùng để xác định sản phẩm nào xuất hiện trong prior).

    Trả về
    -------
    nx.Graph
    """
    print("  Đang xây dựng đồ thị...")

    n_products = spmi_matrix.shape[0]
    n_departments = products_df["department_id"].max() + 1

    graph = nx.Graph()

    # Thêm node product
    product_ids_in_prior = set(prior_df["product_id"].unique())

    for pid in range(n_products):
        product_label = f"product_{pid}"
        in_prior = 1 if pid in product_ids_in_prior else 0
        graph.add_node(product_label, type="product", in_prior=in_prior)

    # Thêm node department
    for did in range(n_departments):
        dept_label = f"dept_{did}"
        graph.add_node(dept_label, type="department")

    # Thêm edges co_purchase từ SPMI matrix
    print(f"  Đang thêm edges co_purchase từ SPMI (SPMI > 0)...")
    edge_count = 0
    spmi_coo = spmi_matrix.tocoo()

    for i, j, w in tqdm(
        zip(spmi_coo.row, spmi_coo.col, spmi_coo.data),
        desc="  Thêm edges co_purchase",
        unit="edges",
        total=spmi_coo.nnz,
    ):
        # Chỉ thêm tam giác trên để tránh trùng lặp trong đồ thị vô hướng
        if i < j and w > 0:
            graph.add_edge(
                f"product_{i}",
                f"product_{j}",
                weight=float(w),
                edge_type="co_purchase",
            )
            edge_count += 1

    print(f"  Đã thêm {edge_count:,} edges co_purchase")

    # Thêm edges belongs_to (product → department)
    print(f"  Đang thêm belongs_to edges...")
    for _, row in products_df.iterrows():
        pid = row["product_id"]
        did = row["department_id"]
        if pid < n_products and did < n_departments:
            graph.add_edge(
                f"product_{pid}",
                f"dept_{did}",
                weight=1.0,
                edge_type="belongs_to",
            )

    print(f"  Đồ thị: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    return graph


# ============================================================================
# PHASE 2: Node2Vec Random Walks
# ============================================================================

def _node2vec_walk(graph, start_node, walk_length, p, q):
    """
    Thực hiện một random walk bắt đầu từ start_node, dùng node2vec strategy.

    Tham số
    ----------
    graph : nx.Graph
    start_node : str
    walk_length : int
    p : float
    q : float

    Trả về
    -------
    list of str (các node label trong walk)
    """
    walk = [start_node]

    # Cần ít nhất 1 bước mới có "node trước"
    if walk_length <= 1:
        return walk

    # Bước đầu tiên: uniform random từ neighbors (chưa có "node trước")
    neighbors = list(graph.neighbors(start_node))
    if not neighbors:
        return walk
    # Sample với trọng số = edge weight
    weights = np.array([graph[start_node][n].get("weight", 1.0) for n in neighbors])
    weights = weights / weights.sum()
    first_step = np.random.choice(neighbors, p=weights)
    walk.append(first_step)

    # Các bước tiếp theo: biased random walk
    for _ in range(2, walk_length):
        t = walk[-2]  # node trước
        v = walk[-1]  # node hiện tại
        neighbors = list(graph.neighbors(v))
        if not neighbors:
            break

        # Tính trọng số α_pq
        probs = []
        for x in neighbors:
            w_vx = graph[v][x].get("weight", 1.0)
            if x == t:
                alpha = 1.0 / p
            elif graph.has_edge(t, x):
                alpha = 1.0
            else:
                alpha = 1.0 / q
            probs.append(w_vx * alpha)

        probs = np.array(probs)
        probs = probs / probs.sum()
        next_node = np.random.choice(neighbors, p=probs)
        walk.append(next_node)

    return walk


def _generate_walks(graph, num_walks, walk_length, p=1.0, q=1.0):
    """
    Sinh node2vec random walks cho tất cả các node.

    Tham số
    ----------
    graph : nx.Graph
    num_walks : int — số walk mỗi node
    walk_length : int — độ dài mỗi walk
    p : float
    q : float

    Trả về
    -------
    list of list of str
    """
    nodes = list(graph.nodes())
    walks = []

    for node in tqdm(nodes, desc="  Sinh walks", unit="nodes"):
        for _ in range(num_walks):
            walk = _node2vec_walk(graph, node, walk_length, p, q)
            walks.append(walk)

    return walks


# ============================================================================
# PHASE 3: Skip-Gram với Negative Sampling
# ============================================================================

def _build_node_index(nodes):
    """
    Xây dựng ánh xạ node label → index (0..N-1).

    Tham số
    ----------
    nodes : list of str

    Trả về
    -------
    dict: {label: index}
    dict: {index: label}
    int: tổng số nodes
    """
    node_to_idx = {}
    idx_to_node = {}
    for idx, node in enumerate(nodes):
        node_to_idx[node] = idx
        idx_to_node[idx] = node
    return node_to_idx, idx_to_node, len(nodes)


def _skipgram_train(walks, n_nodes, dimensions, window, negative, epochs, lr):
    """
    Huấn luyện skip-gram với negative sampling cho tất cả walks.

    Tham số
    ----------
    walks : list of list of int (các walks đã convert sang index)
    n_nodes : int — tổng số node (bao gồm cả product + department)
    dimensions : int — kích thước embedding
    window : int — context window size
    negative : int — số negative samples mỗi positive
    epochs : int — số epoch training
    lr : float — learning rate ban đầu

    Trả về
    -------
    numpy array (n_nodes × dimensions) — embedding matrix
    """
    # Khởi tạo embeddings ngẫu nhiên
    # Dùng 2 ma trận: W_in (center) và W_out (context), giống word2vec
    W_in = np.random.uniform(-0.5 / dimensions, 0.5 / dimensions, (n_nodes, dimensions)).astype(np.float32)
    W_out = np.random.uniform(-0.5 / dimensions, 0.5 / dimensions, (n_nodes, dimensions)).astype(np.float32)

    # Xây dựng noise distribution từ tần suất node trong walks
    freq = np.zeros(n_nodes, dtype=np.float64)
    for walk in walks:
        for node in walk:
            freq[node] += 1

    # Unigram distribution ^ power
    freq_pow = freq ** 0.75
    noise_dist = freq_pow / freq_pow.sum()

    # Đếm tổng số training pairs (để hiển thị progress)
    total_pairs = 0
    for walk in walks:
        if len(walk) < 2:
            continue
        total_pairs += (len(walk) - 1) * min(window, len(walk) - 1) * 2  # ước lượng

    print(f"  [Skip-Gram] Tổng nodes: {n_nodes}, dim={dimensions}, "
          f"negative={negative}, epochs={epochs}")
    print(f"  [Skip-Gram] ~{total_pairs:,} training pairs (ước lượng)")

    # Training
    total_steps = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        pairs_count = 0

        # Shuffle walks
        walk_indices = np.random.permutation(len(walks))

        for walk_idx in tqdm(walk_indices, desc=f"  Epoch {epoch+1}/{epochs}", unit="walks"):
            walk = walks[walk_idx]
            if len(walk) < 2:
                continue

            for center_pos in range(len(walk)):
                center = walk[center_pos]

                # Context window
                left = max(0, center_pos - window)
                right = min(len(walk), center_pos + window + 1)

                for ctx_pos in range(left, right):
                    if ctx_pos == center_pos:
                        continue
                    context = walk[ctx_pos]

                    # ---- Positive sample ----
                    # grad cho center: (σ - 1) * W_out[context]
                    # grad cho context: (σ - 1) * W_in[center]
                    dot = np.dot(W_in[center], W_out[context])
                    sig = 1.0 / (1.0 + np.exp(-dot))  # sigmoid
                    grad = sig - 1.0  # cho positive sample

                    # Update
                    W_in[center] -= lr * grad * W_out[context]
                    W_out[context] -= lr * grad * W_in[center]

                    # Loss: -log(sig)
                    epoch_loss += -np.log(max(sig, 1e-15))

                    # ---- Negative samples ----
                    neg_samples = np.random.choice(
                        n_nodes, size=negative, p=noise_dist, replace=False
                    )

                    for neg in neg_samples:
                        if neg == center or neg == context:
                            continue

                        dot_neg = np.dot(W_in[center], W_out[neg])
                        sig_neg = 1.0 / (1.0 + np.exp(-dot_neg))
                        grad_neg = sig_neg  # cho negative: grad = σ - 0

                        W_in[center] -= lr * grad_neg * W_out[neg]
                        W_out[neg] -= lr * grad_neg * W_in[center]

                        epoch_loss += -np.log(max(1.0 - sig_neg, 1e-15))

                    pairs_count += 1
                    total_steps += 1

        avg_loss = epoch_loss / max(pairs_count, 1)
        print(f"  Epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.4f}, lr: {lr:.6f}")

        # Giảm learning rate sau mỗi epoch
        lr *= 0.95

    # Trả về W_in làm embedding chính (dùng center embedding)
    return W_in


# ============================================================================
# PHASE 4: Tích hợp — train_node2vec
# ============================================================================

def train_node2vec(graph, dimensions=128, walk_length=20, num_walks=200,
                   window=KG_WINDOW, p=KG_P, q=KG_Q, epochs=KG_EPOCHS, negative=KG_NEGATIVE,
                   lr=KG_LR):
    """
    Huấn luyện node2vec embeddings trên đồ thị.

    Pipeline:
      1. Sinh random walks với node2vec strategy
      2. Ánh xạ node label → index
      3. Skip-gram với negative sampling

    Tham số
    ----------
    graph : nx.Graph
    dimensions : int — Kích thước embedding.
    walk_length : int — Độ dài mỗi random walk.
    num_walks : int — Số walks mỗi node.
    window : int — Context window size.
    p : float — Return parameter.
    q : float — In-out parameter.
    epochs : int — Số epoch training skip-gram.
    negative : int — Số negative samples mỗi positive.
    lr : float — Learning rate.

    Trả về
    -------
    tuple: (model_dict, embeddings_matrix)
        model_dict: dict chứa W_in (để tương thích với code cũ)
        embeddings_matrix: numpy array (n_products x dimensions)
            Chỉ chứa product nodes, index theo product_id.
    """
    print(f"  Đang huấn luyện node2vec: dim={dimensions}, walk_len={walk_length}, "
          f"num_walks={num_walks}, epochs={epochs}...")

    # Bước 1: Sinh walks
    print(f"  [1/3] Đang sinh random walks...")
    walks = _generate_walks(graph, num_walks, walk_length, p=p, q=q)
    print(f"  Đã sinh {len(walks):,} walks")

    # Bước 2: Ánh xạ nodes sang indices
    print(f"  [2/3] Đang xây dựng node index...")
    all_nodes = list(graph.nodes())
    node_to_idx, idx_to_node, n_nodes = _build_node_index(all_nodes)

    # Convert walks từ label → index
    walks_idx = []
    for walk in walks:
        walk_idx = [node_to_idx[node] for node in walk if node in node_to_idx]
        if len(walk_idx) > 1:
            walks_idx.append(walk_idx)

    print(f"  Đã convert {len(walks_idx):,} walks sang indices")

    # Bước 3: Skip-gram training
    print(f"  [3/3] Đang huấn luyện skip-gram với negative sampling...")
    W_in = _skipgram_train(
        walks_idx, n_nodes, dimensions, window=window,
        negative=negative, epochs=epochs, lr=lr
    )

    # Trích xuất product embeddings (chỉ các node product_<id>)
    n_products = max(
        int(node.replace("product_", ""))
        for node in graph.nodes()
        if node.startswith("product_")
    ) + 1

    embeddings = np.zeros((n_products, dimensions), dtype=np.float32)

    for pid in range(n_products):
        label = f"product_{pid}"
        if label in node_to_idx:
            idx = node_to_idx[label]
            embeddings[pid] = W_in[idx].astype(np.float32)

    print(f"  Embeddings shape: {embeddings.shape}")

    # Trả về model dict để tương thích với code cũ
    model_dict = {"W_in": W_in, "node_to_idx": node_to_idx}
    return model_dict, embeddings


# ============================================================================
# PHASE 5: Cosine Similarity
# ============================================================================

def compute_kg_similarity(embeddings, top_k=KG_TOP_K, chunk_size=KG_CHUNK_SIZE):
    """
    Tính cosine similarity giữa các product embeddings.

    L2 normalize → chunked dot product → top-K.

    Tham số
    ----------
    embeddings : numpy array (n_products x dimensions)
    top_k : int
        Chỉ giữ top-K similar products mỗi dòng.
    chunk_size : int

    Trả về
    -------
    csr_matrix (n_products x n_products)
    """
    print("  Đang tính cosine similarity...")
    n = embeddings.shape[0]

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Tránh chia cho 0 với cold-start products
    embeddings_norm = embeddings / norms

    sim_lil = lil_matrix((n, n), dtype=np.float32)

    for start in tqdm(range(0, n, chunk_size), desc="  Tính chunks"):
        end = min(start + chunk_size, n)
        chunk = embeddings_norm[start:end]  # (chunk_size × dim)
        sim_chunk = chunk @ embeddings_norm.T  # (chunk_size × n)

        for i_local, row_idx in enumerate(range(start, end)):
            row = sim_chunk[i_local]
            # Đặt self-similarity = 0
            row[row_idx] = 0
            # Lấy top-K
            top_indices = np.argpartition(row, -top_k)[-top_k:]
            top_values = row[top_indices]
            # Sắp xếp giảm dần
            sorted_idx = np.argsort(top_values)[::-1]
            top_indices = top_indices[sorted_idx]
            top_values = top_values[sorted_idx]
            # Chỉ giữ giá trị dương
            positive = top_values > 0
            if positive.any():
                sim_lil[row_idx, top_indices[positive]] = top_values[positive].astype(np.float32)

    sim_csr = sim_lil.tocsr()
    print(f"  KG similarity: {sim_csr.shape}, non-zero: {sim_csr.nnz:,}")

    return sim_csr


# ============================================================================
# PHASE 6: Đánh giá + Tuning
# ============================================================================

def evaluate_kg_in_sample(sim_matrix, train_gt_df, ks=KG_EVAL_KS):
    """
    Đánh giá in-sample KG similarity trên tập train.
    Cùng giao thức leave-one-out mỗi sản phẩm như SPMI evaluation.

    Tham số
    ----------
    sim_matrix : csr_matrix
    train_gt_df : pd.DataFrame
    ks : tuple of int

    Trả về
    -------
    dict: {f"recall@{k}": value}
    """
    print(f"\n  Đánh giá KG in-sample ({len(train_gt_df):,} interactions)...")

    order_groups = train_gt_df.groupby("order_id")["product_id"].apply(list)

    hits = {k: 0.0 for k in ks}
    total_queries = 0

    for products in tqdm(
        order_groups.values,
        desc="  Đánh giá KG",
        unit="orders",
        total=len(order_groups),
    ):
        n = len(products)
        if n < 2:
            continue

        products_set = set(products)

        for i in range(n):
            query = products[i]
            ground_truth = products_set - {query}

            row = sim_matrix[query]
            if row.nnz == 0:
                continue

            row_data = row.data
            row_indices = row.indices
            sorted_idx = np.argsort(row_data)[::-1]

            max_k = max(ks)
            top_indices = row_indices[sorted_idx[:max_k]]

            for k in ks:
                top_k_set = set(top_indices[:k].tolist())
                if top_k_set & ground_truth:
                    hits[k] += 1

            total_queries += 1

    results = {}
    print(f"\n  Tổng queries: {total_queries:,}")
    for k in ks:
        recall = hits[k] / total_queries if total_queries > 0 else 0
        results[f"recall@{k}"] = round(recall, 6)
        print(f"  Recall@{k}: {recall:.4f}")

    return results


def tune_kg_params(graph, train_gt_df,
                   walk_lengths=KG_WALK_LENGTHS,
                   dimensions_list=KG_DIMENSIONS_LIST,
                   num_walks_list=KG_NUM_WALKS_LIST):
    """
    Grid search node2vec hyperparameters trên tập train.

    Tham số
    ----------
    graph : nx.Graph
    train_gt_df : pd.DataFrame
    walk_lengths : tuple
    dimensions_list : tuple
    num_walks_list : tuple

    Trả về
    -------
    tuple: (best_params, best_embeddings, best_sim, all_results)
    """
    print("=" * 50)
    print("Tuning KG (node2vec) parameters")
    print("=" * 50)

    best_score = -1
    best_params = None
    best_embeddings = None
    best_sim = None
    all_results = []

    total_combos = len(walk_lengths) * len(dimensions_list) * len(num_walks_list)
    combo_idx = 0

    for walk_length in walk_lengths:
        for dimensions in dimensions_list:
            for num_walks in num_walks_list:
                combo_idx += 1
                print(f"\n--- [{combo_idx}/{total_combos}] walk_len={walk_length}, "
                      f"dim={dimensions}, num_walks={num_walks} ---")

                # Huấn luyện
                _, embeddings = train_node2vec(
                    graph,
                    dimensions=dimensions,
                    walk_length=walk_length,
                    num_walks=num_walks,
                )

                # Tính similarity
                sim = compute_kg_similarity(embeddings, top_k=KG_TOP_K)

                # Đánh giá trên train
                metrics = evaluate_kg_in_sample(sim, train_gt_df)
                score = metrics.get("recall@5", 0)

                result = {
                    "walk_length": walk_length,
                    "dimensions": dimensions,
                    "num_walks": num_walks,
                    "metrics": metrics,
                }
                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "walk_length": walk_length,
                        "dimensions": dimensions,
                        "num_walks": num_walks,
                    }
                    best_embeddings = embeddings
                    best_sim = sim
                    print(f"  >>> Kết quả tốt nhất mới! recall@5 = {score:.4f}")

    print(f"\nTham số tốt nhất: {best_params}")
    print(f"Recall@5 tốt nhất: {best_score:.4f}")

    return best_params, best_embeddings, best_sim, all_results


# ============================================================================
# PHASE 7: Pipeline chính
# ============================================================================

def build_kg_model(spmi_matrix, products_df, prior_df, train_gt_df):
    """
    Pipeline KG đầy đủ: đồ thị → node2vec → similarity → tune → lưu.

    Tham số
    ----------
    spmi_matrix : csr_matrix
    products_df : pd.DataFrame
    prior_df : pd.DataFrame
    train_gt_df : pd.DataFrame

    Trả về
    -------
    tuple: (best_params, embeddings, similarity, tuning_results)
    """
    print("=" * 50)
    print("Xây dựng Knowledge Graph (KG) Model")
    print("=" * 50)

    # Bước 1: Xây dựng đồ thị
    print("\n[1/3] Đang xây dựng đồ thị từ SPMI + product metadata...")
    graph = build_graph(spmi_matrix, products_df, prior_df)

    # Bước 2: Tune node2vec trên train
    print("\n[2/3] Đang tuning node2vec hyperparameters...")
    best_params, embeddings, sim_matrix, tuning_results = tune_kg_params(
        graph, train_gt_df
    )

    # Bước 3: Hoàn tất
    print(f"\n[3/3] KG model hoàn tất với tham số tốt nhất: {best_params}")

    return best_params, embeddings, sim_matrix, tuning_results


def save_model(best_params, embeddings, sim_matrix, tuning_results):
    """
    Lưu KG model outputs.

    Tham số
    ----------
    best_params : dict
    embeddings : numpy array
    sim_matrix : csr_matrix
    tuning_results : list
    """
    print("\nĐang lưu KG model outputs...")

    np.save(MODELS_DIR / KG_EMBEDDINGS_FILE, embeddings)
    print(f"  Đã lưu: models/{KG_EMBEDDINGS_FILE}")

    with open(MODELS_DIR / KG_BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Đã lưu: models/{KG_BEST_PARAMS_FILE}")

    save_npz(MODELS_DIR / KG_SIMILARITY_FILE, sim_matrix)
    print(f"  Đã lưu: models/{KG_SIMILARITY_FILE}")

    # Lưu thêm tuning results để tham khảo
    with open(MODELS_DIR / KG_TUNING_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"  Đã lưu: models/{KG_TUNING_RESULTS_FILE}")

    print("\nKG model hoàn tất!")


if __name__ == "__main__":
    from src.utils.data_loader import load_products, load_order_products, load_train_test_split

    # Tải dữ liệu
    print("Đang tải dữ liệu...")
    products_df = load_products()
    prior_df = load_order_products("prior")
    train_gt_df, _ = load_train_test_split()

    # Tải SPMI matrix từ bước trước
    print("Đang tải SPMI matrix...")
    spmi_matrix = load_npz(MODELS_DIR / "spmi_matrix.npz")

    # Xây dựng model
    best_params, embeddings, sim_matrix, tuning_results = build_kg_model(
        spmi_matrix, products_df, prior_df, train_gt_df
    )

    # Lưu
    save_model(best_params, embeddings, sim_matrix, tuning_results)