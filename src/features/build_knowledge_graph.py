"""
Knowledge Graph — node2vec embeddings từ graph product-department + SPMI
Tìm sản phẩm LIÊN QUAN qua đồ thị
"""
import gc, json
import numpy as np
import networkx as nx
from collections import Counter
from scipy.sparse import load_npz, save_npz, csr_matrix, lil_matrix
from tqdm import tqdm

from src.config import MODELS_DIR, KG_DIM, KG_WALK_LENGTH, KG_NUM_WALKS, KG_EPOCHS

EMB_FILE = MODELS_DIR / "kg_embeddings.npy"
SIM_FILE = MODELS_DIR / "kg_similarity.npz"

def build_graph(spmi, products_df):
    """Đồ thị product-product (SPMI edges) + product-department (belongs_to)"""
    print("\n  [KG] Building graph ...")
    G = nx.Graph()
    n = spmi.shape[0]
    # Product nodes
    G.add_nodes_from([f"p{i}" for i in range(n)], type="product")
    # SPMI edges
    coo = spmi.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i < j and w > 0:
            G.add_edge(f"p{i}", f"p{j}", weight=float(w))
    # Department nodes + belongs_to edges
    for _, row in products_df.iterrows():
        pid, did = int(row["product_id"]), int(row["department_id"])
        if pid < n:
            G.add_edge(f"p{pid}", f"d{did}", weight=1.0)
    print(f"  [KG] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

def train_node2vec(G, n_products, dim=KG_DIM, walk_len=KG_WALK_LENGTH, n_walks=KG_NUM_WALKS, epochs=KG_EPOCHS):
    """Skip-gram negative sampling — random walks đơn giản, không bias p/q"""
    print(f"  [KG] Training node2vec dim={dim} ...")
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    n2i = {n: i for i, n in enumerate(nodes)}

    # Random walks
    walks = []
    for start in tqdm(nodes, desc="  Walks"):
        for _ in range(n_walks):
            walk = [start]
            for _ in range(walk_len - 1):
                nb = list(G.neighbors(walk[-1]))
                if not nb:
                    break
                w = np.array([G[walk[-1]][n].get("weight", 1.0) for n in nb], dtype=float)
                w /= w.sum()
                walk.append(np.random.choice(nb, p=w))
            walks.append([n2i[n] for n in walk if n in n2i])

    # Skip-gram init
    W_in = np.random.uniform(-0.5 / dim, 0.5 / dim, (n_nodes, dim)).astype(np.float32)
    W_out = np.random.uniform(-0.5 / dim, 0.5 / dim, (n_nodes, dim)).astype(np.float32)

    # Noise distribution
    freq = Counter()
    for w in walks:
        for nid in w:
            freq[nid] += 1
    noise = np.array([freq[i] ** 0.75 for i in range(n_nodes)], dtype=float)
    noise /= noise.sum()

    lr = 0.025
    for epoch in range(epochs):
        loss = 0.0
        for walk in tqdm(walks, desc=f"  Epoch {epoch+1}/{epochs}"):
            for cpos in range(len(walk)):
                center = walk[cpos]
                left = max(0, cpos - dim // 10)
                right = min(len(walk), cpos + dim // 10 + 1)
                for ctx_pos in range(left, right):
                    if ctx_pos == cpos:
                        continue
                    ctx = walk[ctx_pos]
                    # Positive
                    dot = np.dot(W_in[center], W_out[ctx])
                    sig = 1.0 / (1.0 + np.exp(-dot))
                    grad = sig - 1.0
                    W_in[center] -= lr * grad * W_out[ctx]
                    W_out[ctx] -= lr * grad * W_in[center]
                    loss += -np.log(max(sig, 1e-15))
                    # Negative samples
                    negs = np.random.choice(n_nodes, size=5, p=noise, replace=False)
                    for neg in negs:
                        if neg == center or neg == ctx:
                            continue
                        dot_n = np.dot(W_in[center], W_out[neg])
                        sig_n = 1.0 / (1.0 + np.exp(-dot_n))
                        grad_n = sig_n
                        W_in[center] -= lr * grad_n * W_out[neg]
                        W_out[neg] -= lr * grad_n * W_in[center]
                        loss += -np.log(max(1.0 - sig_n, 1e-15))
        lr *= 0.95
        print(f"    loss={loss:.4f}")

    # Trích xuất product embeddings
    # n_products đã được truyền vào
    emb = np.zeros((n_products, dim), dtype=np.float32)
    for pid in range(n_products):
        label = f"p{pid}"
        if label in n2i:
            emb[pid] = W_in[n2i[label]]

    del walks, n2i, W_in, W_out; gc.collect()
    return emb

def compute_similarity(emb, top_k=100, chunk=1000):
    """Cosine similarity chunked — chỉ giữ top-K"""
    n = emb.shape[0]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = emb / norms
    sim = lil_matrix((n, n), dtype=np.float32)
    for start in tqdm(range(0, n, chunk), desc="  Similarity"):
        end = min(start + chunk, n)
        chunk_sim = emb_norm[start:end] @ emb_norm.T
        for i_local, row_idx in enumerate(range(start, end)):
            row = chunk_sim[i_local]
            row[row_idx] = 0
            if top_k < n:
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
    np.save(EMB_FILE, emb)
    save_npz(SIM_FILE, sim)
    print(f"  [KG] Saved: {EMB_FILE}, {SIM_FILE}")

def load():
    return np.load(EMB_FILE), load_npz(SIM_FILE)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(MODELS_DIR.parent))
    from src.data_loader import load_products, load_prior
    products = load_products()
    prior = load_prior()
    spmi = load_npz(MODELS_DIR / "spmi_matrix.npz")
    G = build_graph(spmi, products)
    emb = train_node2vec(G, spmi.shape[0])
    sim = compute_similarity(emb)
    save(emb, sim)