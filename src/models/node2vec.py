"""
Node2Vec: Graph-based embedding.
Xây đồ thị sản phẩm dựa trên co-occurrence, học embedding qua random walk.
"""
import os
import json
import random
from itertools import combinations
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from src.config import (
    N2V_EMBEDDING_DIM, N2V_WALK_LENGTH, N2V_NUM_WALKS,
    N2V_P, N2V_Q, N2V_WORKERS, N2V_WINDOW, N2V_EDGE_THRESHOLD, N2V_TOP_K,
    RANDOM_SEED
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
        self.graph = None          # adjacency list {node: [(neighbor, weight), ...]}
        self.model = None          # Word2Vec model sẽ train sau
        self.embeddings = None     # numpy array (n_products, embedding_dim)
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self._walks = None         # cached walks
    
    def fit(self, order_products: pd.DataFrame, products_df: pd.DataFrame):
        """
        Build co-occurrence graph + random walk + train Word2Vec.
        
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
        
        # --- Bước 1: Đếm co-occurrence ---
        print(f"  Đang đếm co-occurrence (threshold={self.params['edge_threshold']})...")
        pair_counts = defaultdict(int)
        
        grouped = order_products.groupby('order_id')['product_id']
        for order_id, group in tqdm(grouped, desc="  Duyệt orders", unit="order"):
            items = [self.product_id_to_idx[pid] for pid in group
                     if pid in self.product_id_to_idx]
            for a, b in combinations(items, 2):
                pair_counts[(a, b)] += 1
        
        # --- Bước 2: Xây adjacency list ---
        print("  Xây adjacency list...")
        graph = defaultdict(list)
        edge_count = 0
        for (a, b), cnt in pair_counts.items():
            if cnt >= self.params['edge_threshold']:
                graph[a].append((b, cnt))
                graph[b].append((a, cnt))
                edge_count += 1
        
        self.graph = dict(graph)
        print(f"  Số nodes: {len(self.graph)}")
        print(f"  Số edges: {edge_count}")
        
        # Cache neighbor sets cho _is_common_neighbor
        self.neighbor_sets = {
            node: set(n for n, w in neighbors)
            for node, neighbors in self.graph.items()
        }
        
        # --- Bước 3: Random Walk ---
        print(f"  Đang random walk (num_walks={self.params['num_walks']}, "
              f"walk_length={self.params['walk_length']})...")
        
        random.seed(RANDOM_SEED)
        walks = self._generate_walks()
        self._walks = walks
        
        # Convert walks sang định dạng string cho Word2Vec
        sentences = [[str(node) for node in walk] for walk in walks]
        
        # --- Bước 4: Train Word2Vec ---
        print(f"  Đang train Word2Vec (dim={self.params['embedding_dim']})...")
        from gensim.models import Word2Vec
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.params['embedding_dim'],
            window=self.params['window'],
            min_count=1,  # giữ tất cả nodes
            negative=10,
            epochs=20,
            workers=self.params['workers'],
            sg=1,
            seed=RANDOM_SEED,
        )
        
        # Extract embeddings
        n_nodes = len(self.graph)
        self.embeddings = np.zeros((n_products, self.params['embedding_dim']))
        for node in self.graph.keys():
            pid = self.idx_to_product_id[node]
            pid_str = str(pid)
            if pid_str in self.model.wv:
                self.embeddings[node] = self.model.wv[pid_str]
        
        # Cache embedding norms để dùng trong recommend
        self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        self._embedding_norms[self._embedding_norms == 0] = 1e-9
        
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print("Node2Vec: Fit hoàn tất.")
    
    def _generate_walks(self):
        """
        Sinh random walks trên graph.
        
        Returns:
            list of list: [ [node1, node2, ...], ... ]
        """
        nodes = list(self.graph.keys())
        walks = []
        
        for _ in tqdm(range(self.params['num_walks']), desc="  Walks"):
            random.shuffle(nodes)
            for start_node in nodes:
                walk = self._random_walk(start_node)
                walks.append(walk)
        
        return walks
    
    def _random_walk(self, start_node):
        """
        Sinh 1 random walk bắt đầu từ start_node với tham số p, q.
        
        Node2Vec random walk: kiểm soát BFS/DFS qua p và q.
          - p (return): xác suất quay lại node trước
          - q (in-out): xác suất đi xa (DFS) vs ở gần (BFS)
        """
        walk = [start_node]
        p = self.params['p']
        q = self.params['q']
        
        for _ in range(self.params['walk_length'] - 1):
            cur = walk[-1]
            neighbors = self.graph.get(cur, [])
            if not neighbors:
                break
            
            if len(walk) == 1:
                # Bước đầu tiên: random uniform
                next_node = random.choice([n for n, w in neighbors])
            else:
                prev = walk[-2]
                next_node = self._alias_sample(cur, prev, neighbors, p, q)
            
            walk.append(next_node)
        
        return walk
    
    def _alias_sample(self, cur, prev, neighbors, p, q):
        """
        Lấy mẫu node kế tiếp dựa trên p và q.
        
        Args:
            cur: node hiện tại
            prev: node trước đó
            neighbors: list [(neighbor, weight)]
            p: return parameter (xác suất quay lại node trước)
            q: in-out parameter (xác suất đi xa vs ở gần)
        
        Returns:
            next_node: int
        """
        weights = []
        for neighbor, w in neighbors:
            if neighbor == prev:
                # Quay lại node trước
                weights.append(w / p)
            elif self._is_common_neighbor(cur, prev, neighbor):
                # Node chung của cur và prev (BFS)
                weights.append(w)
            else:
                # Node xa (DFS)
                weights.append(w / q)
        
        total = sum(weights)
        if total == 0:
            return random.choice([n for n, w in neighbors])
        
        r = random.random() * total
        cumsum = 0
        for (neighbor, w), weight in zip(neighbors, weights):
            cumsum += weight
            if r <= cumsum:
                return neighbor
        
        return neighbors[-1][0]
    
    def _is_common_neighbor(self, cur, prev, neighbor):
        """Kiểm tra neighbor có phải là common neighbor của cur và prev không."""
        return neighbor in self.neighbor_sets.get(prev, set())
    
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
            'product_id_to_idx': self.product_id_to_idx,
            'idx_to_product_id': {str(k): v for k, v in self.idx_to_product_id.items()},
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
        
        # Rebuild neighbor_sets cache
        self.neighbor_sets = {
            node: set(n for n, w in neighbors)
            for node, neighbors in self.graph.items()
        }
        
        print(f"Node2Vec: Đã load từ {path}")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Số nodes trong graph: {len(self.graph)}")