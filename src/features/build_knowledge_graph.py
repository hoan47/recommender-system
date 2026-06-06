"""
Knowledge Graph — Node2Vec embeddings từ đồ thị product-department

Xây dựng đồ thị với 2 loại node:
  - Product nodes (p0..pN): đại diện cho sản phẩm
  - Department nodes (d0..dM): đại diện cho ngành hàng

Edge gồm:
  1. co_purchase: product-product, weight = SPMI score (từ build_spmi)
     Chỉ giữ cặp có SPMI > 0 để lọc nhiễu
  2. belongs_to: product-department, weight = 1.0 (từ products.csv)

Sau đó dùng node2vec (random walks + skip-gram negative sampling)
để học product embeddings. Cuối cùng tính cosine similarity giữa
các product embeddings để tìm sản phẩm liên quan qua đồ thị.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gc
import numpy as np
import networkx as nx
from collections import Counter
from scipy.sparse import load_npz, save_npz, lil_matrix
from tqdm import tqdm

from src.config import MODELS_DIR, KG_DIM, KG_WALK_LENGTH, KG_NUM_WALKS, KG_EPOCHS

# File lưu embeddings và similarity matrix
EMB_FILE = MODELS_DIR / "kg_embeddings.npy"
SIM_FILE = MODELS_DIR / "kg_similarity.npz"

def build_graph(spmi, products_df):
    """
    Xây dựng đồ thị NetworkX từ SPMI matrix và product metadata.
    - SPMI edges: product-product (chỉ giữ tam giác trên để tránh trùng)
    - belongs_to edges: product-department (từ products.csv)
    
    Tham số:
        spmi: csr_matrix — ma trận SPMI từ build_spmi
        products_df: DataFrame — product_id, department_id
        
    Trả về:
        G: nx.Graph — đồ thị vô hướng
    """
    print("\n  [KG] Building graph ...")
    G = nx.Graph()
    n = spmi.shape[0]
    
    # Thêm product nodes (p0..pn-1) với type="product"
    G.add_nodes_from([f"p{i}" for i in range(n)], type="product")
    
    # Thêm edges co_purchase từ SPMI matrix (chỉ tam giác trên i<j)
    coo = spmi.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i < j and w > 0:
            G.add_edge(f"p{i}", f"p{j}", weight=float(w))
    
    # Thêm department nodes (d0..dm) + edges belongs_to
    for _, row in products_df.iterrows():
        pid, did = int(row["product_id"]), int(row["department_id"])
        if pid < n:
            G.add_edge(f"p{pid}", f"d{did}", weight=1.0)
    
    print(f"  [KG] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

def train_node2vec(G, n_products, dim=KG_DIM, walk_len=KG_WALK_LENGTH, n_walks=KG_NUM_WALKS, epochs=KG_EPOCHS):
    """
    Huấn luyện node2vec embeddings.
    
    Random walks: mỗi node sinh n_walks walks, mỗi walk dài walk_len.
    Bước tiếp theo được chọn dựa trên edge weight (trọng số càng cao càng dễ chọn).
    Không dùng bias p/q (node2vec chuẩn) để giữ code đơn giản.
    
    Skip-gram với negative sampling:
    - Center embedding (W_in) và context embedding (W_out)
    - Positive: cặp (center, context) trong cùng window
    - Negative: sample ngẫu nhiên 5 node theo noise distribution (freq^0.75)
    
    Tham số:
        G: nx.Graph — đồ thị đầu vào
        n_products: int — số lượng product (để tách product embeddings từ node embeddings)
        dim: int — kích thước embedding vector
        walk_len: int — độ dài mỗi random walk
        n_walks: int — số walk mỗi node
        epochs: int — số epoch huấn luyện skip-gram
        
    Trả về:
        emb: numpy array (n_products x dim) — embeddings cho mỗi product
    """
    print(f"  [KG] Training node2vec dim={dim} ...")
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    n2i = {n: i for i, n in enumerate(nodes)}  # Node label → index

    # Bước 1: Sinh random walks
    walks = []
    for start in tqdm(nodes, desc="  Walks"):
        for _ in range(n_walks):
            walk = [start]
            for _ in range(walk_len - 1):
                nb = list(G.neighbors(walk[-1]))
                if not nb:
                    break
                # Chọn neighbor theo edge weight (weighted random)
                w = np.array([G[walk[-1]][n].get("weight", 1.0) for n in nb], dtype=float)
                w /= w.sum()
                walk.append(np.random.choice(nb, p=w))
            # Convert node label → index để huấn luyện
            walks.append([n2i[n] for n in walk if n in n2i])

    # Bước 2: Khởi tạo embeddings ngẫu nhiên (Xavier init style)
    W_in = np.random.uniform(-0.5 / dim, 0.5 / dim, (n_nodes, dim)).astype(np.float32)
    W_out = np.random.uniform(-0.5 / dim, 0.5 / dim, (n_nodes, dim)).astype(np.float32)

    # Bước 3: Noise distribution cho negative sampling (freq^0.75)
    freq = Counter()
    for w in walks:
        for nid in w:
            freq[nid] += 1
    noise = np.array([freq[i] ** 0.75 for i in range(n_nodes)], dtype=float)
    noise /= noise.sum()

    # Bước 4: Huấn luyện skip-gram
    lr = 0.025  # Learning rate ban đầu
    for epoch in range(epochs):
        loss = 0.0
        for walk in tqdm(walks, desc=f"  Epoch {epoch+1}/{epochs}"):
            for cpos in range(len(walk)):
                center = walk[cpos]
                # Context window: dim//10 là kích thước window
                left = max(0, cpos - dim // 10)
                right = min(len(walk), cpos + dim // 10 + 1)
                for ctx_pos in range(left, right):
                    if ctx_pos == cpos:
                        continue
                    ctx = walk[ctx_pos]
                    
                    # Positive sample: (center, context) → sigmoid dot product
                    dot = np.dot(W_in[center], W_out[ctx])
                    sig = 1.0 / (1.0 + np.exp(-dot))
                    grad = sig - 1.0  # Gradient cho positive
                    
                    # Update embeddings
                    W_in[center] -= lr * grad * W_out[ctx]
                    W_out[ctx] -= lr * grad * W_in[center]
                    loss += -np.log(max(sig, 1e-15))
                    
                    # Negative samples: 5 node ngẫu nhiên không phải center/context
                    negs = np.random.choice(n_nodes, size=5, p=noise, replace=False)
                    for neg in negs:
                        if neg == center or neg == ctx:
                            continue
                        dot_n = np.dot(W_in[center], W_out[neg])
                        sig_n = 1.0 / (1.0 + np.exp(-dot_n))
                        grad_n = sig_n  # Gradient cho negative
                        
                        W_in[center] -= lr * grad_n * W_out[neg]
                        W_out[neg] -= lr * grad_n * W_in[center]
                        loss += -np.log(max(1.0 - sig_n, 1e-15))
        lr *= 0.95  # Giảm learning rate sau mỗi epoch
        print(f"    loss={loss:.4f}")

    # Bước 5: Trích xuất product embeddings (bỏ department nodes)
    emb = np.zeros((n_products, dim), dtype=np.float32)
    for pid in range(n_products):
        label = f"p{pid}"
        if label in n2i:
            emb[pid] = W_in[n2i[label]].astype(np.float32)

    del walks, n2i, W_in, W_out; gc.collect()
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
    G = build_graph(spmi, products)
    emb = train_node2vec(G, spmi.shape[0])
    sim = compute_similarity(emb)
    save(emb, sim)