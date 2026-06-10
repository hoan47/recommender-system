"""
Node2Vec: Graph-based embedding.
Xây đồ thị sản phẩm dựa trên co-occurrence, học embedding qua random walk.

Cải tiến Numba:
  - Đếm co-occurrence pairs: count_pairs_numba()
  - Xây adjacency CSR: _build_adjacency_csr()
  - Random walk + alias sampling: generate_walks_numba()
"""
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.config import (
    N2V_EMBEDDING_DIM, N2V_WALK_LENGTH, N2V_NUM_WALKS,
    N2V_P, N2V_Q, N2V_WORKERS, N2V_WINDOW, N2V_EDGE_THRESHOLD, N2V_TOP_K,
    RANDOM_SEED
)
from src.utils._numba_ops import (
    count_pairs_numba,
    _build_adjacency_csr,
    generate_walks_numba,
)


class Node2VecModel:
    """
    Node2Vec: Graph embedding với random walk.
    
    - Node: mỗi sản phẩm
    - Edge: co-occurrence count >= threshold
    - Weight: co-occurrence count (raw)
    """
    
    def __init__(self, embedding_dim=None, walk_length=None, num_walks=None,
                 p=None, q=None, edge_threshold=None, workers=None, window=None):
        self.params = {
            'embedding_dim': embedding_dim if embedding_dim is not None else N2V_EMBEDDING_DIM,
            'walk_length': walk_length if walk_length is not None else N2V_WALK_LENGTH,
            'num_walks': num_walks if num_walks is not None else N2V_NUM_WALKS,
            'p': p if p is not None else N2V_P,
            'q': q if q is not None else N2V_Q,
            'edge_threshold': edge_threshold if edge_threshold is not None else N2V_EDGE_THRESHOLD,
            'workers': workers if workers is not None else N2V_WORKERS,
            'window': window if window is not None else N2V_WINDOW,
        }
        self.graph = None            # adjacency list {node: [(neighbor, weight), ...]}
        self.graph_csr = None        # tuple (indptr, neighbors, weights) — CSR format
        self.model = None            # Word2Vec model
        self.embeddings = None       # numpy array (n_products, embedding_dim)
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self._walks_cache = None     # cache walks numpy array để debug
    
    def fit(self, order_products: pd.DataFrame, products_df: pd.DataFrame):
        """
        Build co-occurrence graph + random walk + train Word2Vec.
        
        Dùng Numba cho:
          1. count_pairs_numba() — đếm pairs nhanh
          2. _build_adjacency_csr() — xây graph CSR
          3. generate_walks_numba() — random walk parallel
        
        Args:
            order_products: DataFrame [order_id, product_id, ...]
            products_df: DataFrame [product_id, ...]
        """
        print("Node2Vec: Bắt đầu fit...")
        
        # Mapping product_id
        all_product_ids = sorted(products_df['product_id'].unique())
        self.product_id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        self.idx_to_product_id = {i: pid for pid, i in self.product_id_to_idx.items()}
        n_products = len(all_product_ids)
        
        # --- Bước 1: Xây CSR-like order_indices cho Numba counting ---
        print(f"  Đang xây order_indices cho {len(order_products):,} records...")
        grouped = order_products.groupby('order_id')['product_id']
        
        total_items = 0
        order_lengths = []
        for order_id, group in tqdm(grouped, desc="  Scan orders", unit="order"):
            items = [self.product_id_to_idx.get(pid, -1) for pid in group]
            items = [x for x in items if x >= 0]
            if items:
                total_items += len(items)
                order_lengths.append(len(items))
        
        order_indices = np.zeros(total_items, dtype=np.int32)
        order_ptr = np.zeros(len(order_lengths) + 1, dtype=np.int32)
        pos = 0
        for o_idx, length in enumerate(order_lengths):
            order_ptr[o_idx] = pos
            pos += length
        order_ptr[-1] = total_items
        
        pos = 0
        for order_id, group in tqdm(grouped, desc="  Fill indices", unit="order"):
            items = [self.product_id_to_idx.get(pid, -1) for pid in group]
            items = [x for x in items if x >= 0]
            n = len(items)
            if n > 0:
                order_indices[pos:pos + n] = items
                pos += n
        
        print(f"  Tổng items: {total_items:,}, tổng orders: {len(order_lengths):,}")
        
        # --- Bước 2: Đếm co-occurrence pairs bằng Numba ---
        print(f"  Đang đếm co-occurrence (threshold={self.params['edge_threshold']})...")
        
        rows, cols, counts = count_pairs_numba(
            order_indices, order_ptr, n_products
        )
        print(f"  Tổng số pairs (raw, unique): {len(rows):,}")
        
        # Lọc edge threshold
        mask = counts >= self.params['edge_threshold']
        pair_rows = rows[mask]
        pair_cols = cols[mask]
        pair_counts = counts[mask]
        print(f"  Sau edge_threshold={self.params['edge_threshold']}: {len(pair_rows):,} edges")
        
        # --- Bước 3: Xây adjacency CSR bằng Numba ---
        print("  Xây graph adjacency...")
        indptr, neighbors, weights = _build_adjacency_csr(
            pair_rows, pair_cols, pair_counts, n_products
        )
        
        # Lưu graph (CSR + dict version cho Python access)
        self.graph_csr = (indptr, neighbors, weights)
        
        # Xây graph dict (needed for save/load và neighbor sets)
        graph = {}
        for node in range(n_products):
            start = indptr[node]
            end = indptr[node + 1]
            if start < end:
                graph[node] = [
                    (neighbors[i], int(weights[i]))
                    for i in range(start, end)
                ]
        self.graph = graph
        
        n_nodes_with_edges = sum(1 for v in graph.values() if len(v) > 0)
        print(f"  Nodes với ít nhất 1 edge: {n_nodes_with_edges:,}")
        print(f"  Tổng edges (undirected): {len(pair_rows):,}")
        
        # --- Bước 4: Random Walk bằng Numba ---
        print(f"  Đang random walk (num_walks={self.params['num_walks']}, "
              f"walk_length={self.params['walk_length']})...")
        
        nodes = np.array(list(self.graph.keys()), dtype=np.int32)
        walks, walk_lengths = generate_walks_numba(
            indptr, neighbors, weights,
            float(self.params['p']), float(self.params['q']),
            int(self.params['walk_length']),
            int(self.params['num_walks']),
            nodes,
            RANDOM_SEED,
        )
        
        self._walks_cache = (walks, walk_lengths)
        print(f"  Tổng walks: {walks.shape[0]:,}")
        print(f"  Walk length trung bình: {walk_lengths.mean():.1f}")
        
        # Convert walks sang sentences cho Word2Vec
        sentences = []
        for i in range(walks.shape[0]):
            wlen = walk_lengths[i]
            if wlen > 1:
                sentence = [str(walks[i, j]) for j in range(wlen)]
                sentences.append(sentence)
        
        print(f"  Sentences (walks có độ dài > 1): {len(sentences):,}")
        
        # --- Bước 5: Train Word2Vec ---
        print(f"  Đang train Word2Vec (dim={self.params['embedding_dim']})...")
        from gensim.models import Word2Vec
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.params['embedding_dim'],
            window=self.params['window'],
            min_count=1,   # giữ tất cả nodes
            negative=10,
            epochs=20,
            workers=self.params['workers'],
            sg=1,
            seed=RANDOM_SEED,
        )
        
        # Extract embeddings
        self.embeddings = np.zeros((n_products, self.params['embedding_dim']))
        for node in self.graph.keys():
            pid = self.idx_to_product_id[node]
            pid_str = str(pid)
            if pid_str in self.model.wv:
                self.embeddings[node] = self.model.wv[pid_str]
        
        # Cache embedding norms
        self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        self._embedding_norms[self._embedding_norms == 0] = 1e-9
        
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print("Node2Vec: Fit hoàn tất.")
    
    def recommend(self, product_id: int, top_k: int = None):
        """
        Cosine similarity trên embedding space — gợi ý sản phẩm mua kèm.
        
        Args:
            product_id: int — ID sản phẩm đầu vào
            top_k: int — số lượng gợi ý tối đa
        
        Returns:
            list (product_id, similarity) — các sản phẩm gợi ý kèm similarity
        """
        if top_k is None:
            top_k = N2V_TOP_K
        
        if product_id not in self.product_id_to_idx:
            return []
        
        idx = self.product_id_to_idx[product_id]
        vec_a = self.embeddings[idx]
        
        if np.linalg.norm(vec_a) == 0:
            return []
        
        # Cosine similarity với tất cả nodes (dùng cached norms)
        similarities = (self.embeddings @ vec_a) / (
            self._embedding_norms * np.linalg.norm(vec_a)
        )
        
        # Bỏ qua chính nó
        similarities[idx] = -1
        
        # Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        result = [
            (self.idx_to_product_id[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]
        
        return result
    
    def save(self, path: str):
        """
        Lưu model ra file (embeddings, metadata, Word2Vec model).
        
        Args:
            path: đường dẫn thư mục đầu ra
        """
        os.makedirs(path, exist_ok=True)
        
        # Lưu embeddings
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        
        # Lưu graph
        graph_serializable = {
            str(k): [(n, w) for n, w in v]
            for k, v in self.graph.items()
        }
        metadata = {
            'params': self.params,
            'graph': graph_serializable,
            'product_id_to_idx': {str(k): int(v) for k, v in self.product_id_to_idx.items()},
            'idx_to_product_id': {str(k): int(v) for k, v in self.idx_to_product_id.items()},
        }
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        # Lưu Word2Vec model
        if self.model is not None:
            self.model.save(os.path.join(path, "word2vec.model"))
        
        print(f"Node2Vec: Đã lưu tại {path}")
    
    def load(self, path: str):
        """
        Load model từ file (embeddings, metadata, Word2Vec model).
        
        Args:
            path: đường dẫn thư mục đã lưu
        """
        # Load embeddings
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        
        # Cache embedding norms
        self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        self._embedding_norms[self._embedding_norms == 0] = 1e-9
        
        # Load metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        self.params = metadata['params']
        self.graph = {
            int(k): [(n, w) for n, w in v]
            for k, v in metadata['graph'].items()
        }
        self.product_id_to_idx = {int(k): int(v) for k, v in metadata['product_id_to_idx'].items()}
        self.idx_to_product_id = {int(k): int(v) for k, v in metadata['idx_to_product_id'].items()}
        
        # Load Word2Vec model
        model_path = os.path.join(path, "word2vec.model")
        if os.path.exists(model_path):
            from gensim.models import Word2Vec
            self.model = Word2Vec.load(model_path)
        
        # Rebuild graph_csr từ graph dict (cần thiết cho Numba nếu sau này gọi lại)
        # Nhưng không bắt buộc vì không cần walk lại sau load
        self.graph_csr = None
        
        print(f"Node2Vec: Đã load từ {path}")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Số nodes trong graph: {len(self.graph)}")