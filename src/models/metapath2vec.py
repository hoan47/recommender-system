"""
Metapath2Vec: Knowledge Graph embedding với Metapath Walk.
Xây dựng Đồ thị Tri thức đa thể (Heterogeneous Knowledge Graph - IKG)
với 3 loại nút (Product, Aisle, Department) và 2 kịch bản Metapath Walk.

Kịch bản 1 (Behavioral): P1 --CO_OCCUR--> P2 --CO_OCCUR--> P3
Kịch bản 2 (Semantic):   P1 --BELONGS_TO--> A --random--> P2

Sau walk → Gensim Skip-gram (phần walk là white-box tự viết).
"""
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.config import (
    MW_EMBEDDING_DIM, MW_WALK_LENGTH, MW_NUM_WALKS,
    MW_WINDOW, MW_NEGATIVE, MW_EPOCHS, MW_WORKERS,
    MW_EDGE_THRESHOLD, MW_TOP_K, MW_METAPATH_BEHAVIORAL_RATIO,
    RANDOM_SEED
)
from src.utils._numba_ops import (
    count_pairs_numba,
    _build_adjacency_csr,
)


class Metapath2VecModel:
    """
    Metapath2Vec: IKG embedding với Metapath Walk có kiểm soát.
    
    Đồ thị IKG:
      - Node: Product (~49K), Aisle (134), Department (21)
      - Edge: CO_OCCUR (P↔P, từ co-occurrence), BELONGS_TO (P→A), PART_OF (A→D)
    
    Metapath Walk (chỉ sinh sequence product IDs):
      1. Behavioral (P→P→P): đi theo cạnh CO_OCCUR
      2. Semantic (P→A→P): nhảy qua Aisle để tiếp cận sản phẩm mới
    
    Skip-gram: dùng Gensim Word2Vec (tầng này không white-box)
    """
    
    def __init__(self, embedding_dim=None, walk_length=None, num_walks=None,
                 edge_threshold=None, window=None, negative=None, epochs=None,
                 workers=None, behavioral_ratio=None):
        self.params = {
            'embedding_dim': embedding_dim if embedding_dim is not None else MW_EMBEDDING_DIM,
            'walk_length': walk_length if walk_length is not None else MW_WALK_LENGTH,
            'num_walks': num_walks if num_walks is not None else MW_NUM_WALKS,
            'edge_threshold': edge_threshold if edge_threshold is not None else MW_EDGE_THRESHOLD,
            'window': window if window is not None else MW_WINDOW,
            'negative': negative if negative is not None else MW_NEGATIVE,
            'epochs': epochs if epochs is not None else MW_EPOCHS,
            'workers': workers if workers is not None else MW_WORKERS,
            'behavioral_ratio': behavioral_ratio if behavioral_ratio is not None else MW_METAPATH_BEHAVIORAL_RATIO,
        }
        # Đồ thị CO_OCCUR (co-occurrence graph)
        self.graph = None            # adjacency list {node: [(neighbor, weight), ...]}
        self.graph_csr = None        # tuple (indptr, neighbors, weights) — CSR format
        # Mapping IKG
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self.product_to_aisle = {}   # product_id → aisle_id
        self.aisle_to_products = {}  # aisle_id → list[product_idx]
        self.model = None            # Word2Vec model
        self.embeddings = None       # numpy array (n_products, embedding_dim)
        self._walks_cache = None     # cache walks để debug
        
    def fit(self, order_products: pd.DataFrame, products_df: pd.DataFrame):
        """
        Xây IKG + Metapath Walk + train Word2Vec.
        
        Args:
            order_products: DataFrame [order_id, product_id, ...]
            products_df: DataFrame [product_id, aisle_id, department_id, ...]
        """
        print("Metapath2Vec: Bắt đầu fit...")
        print(f"  Tham số: dim={self.params['embedding_dim']}, "
              f"walk_length={self.params['walk_length']}, "
              f"num_walks={self.params['num_walks']}, "
              f"behavioral_ratio={self.params['behavioral_ratio']}")
        
        # Mapping product_id → index
        all_product_ids = sorted(products_df['product_id'].unique())
        self.product_id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        self.idx_to_product_id = {i: pid for pid, i in self.product_id_to_idx.items()}
        n_products = len(all_product_ids)
        print(f"  Tổng số product nodes: {n_products:,}")
        
        # --- Xây mapping IKG: Product → Aisle ---
        print("  Xây mapping IKG (Product → Aisle)...")
        for _, row in products_df.iterrows():
            pid = row['product_id']
            aid = row['aisle_id']
            if pid in self.product_id_to_idx:
                pidx = self.product_id_to_idx[pid]
                self.product_to_aisle[pidx] = int(aid)
                if int(aid) not in self.aisle_to_products:
                    self.aisle_to_products[int(aid)] = []
                self.aisle_to_products[int(aid)].append(pidx)
        
        n_aisles_with_products = len(self.aisle_to_products)
        print(f"  Số aisle có sản phẩm: {n_aisles_with_products}")
        
        # --- Xây CSR order_indices cho Numba counting ---
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
        
        # --- Đếm co-occurrence pairs bằng Numba ---
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
        
        # --- Xây adjacency CSR bằng Numba ---
        print("  Xây graph adjacency (CO_OCCUR) bằng Numba...")
        indptr, neighbors, weights = _build_adjacency_csr(
            pair_rows, pair_cols, pair_counts, n_products
        )
        self.graph_csr = (indptr, neighbors, weights)
        
        # Xây graph dict cho Python access
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
        print(f"  Nodes với ít nhất 1 edge CO_OCCUR: {n_nodes_with_edges:,}")
        
        # --- Metapath Walk (2 kịch bản) ---
        print(f"  Đang Metapath Walk (num_walks={self.params['num_walks']}, "
              f"walk_length={self.params['walk_length']})...")
        
        np.random.seed(RANDOM_SEED)
        behavioral_ratio = self.params['behavioral_ratio']
        
        # Nodes có edge CO_OCCUR
        nodes_with_edges = np.array(
            [node for node, v in graph.items() if len(v) > 0],
            dtype=np.int32
        )
        # Nodes có aisle hợp lệ (cho semantic walk)
        nodes_with_aisle = np.array([
            node for node in range(n_products)
            if node in self.product_to_aisle
            and len(self.aisle_to_products.get(self.product_to_aisle[node], [])) >= 2
        ], dtype=np.int32)
        
        # Nếu không có semantic candidates, fallback về behavioral
        if len(nodes_with_aisle) == 0:
            print("  [WARN] Không có semantic candidates → dùng behavioral 100%")
            behavioral_ratio = 1.0
            nodes_with_aisle = nodes_with_edges
        
        n_behavioral = len(nodes_with_edges)
        n_semantic = len(nodes_with_aisle)
        total_walks = max(n_behavioral, n_semantic) * self.params['num_walks']
        walk_length = self.params['walk_length']
        
        print(f"  Nodes cho Behavioral walk: {n_behavioral:,}")
        print(f"  Nodes cho Semantic walk: {n_semantic:,}")
        print(f"  Tổng walks: {total_walks:,}")
        
        # Pre-generate random numbers
        max_degree = int(np.max(np.diff(indptr))) if n_behavioral > 0 else 1
        rand_matrix = np.random.randint(
            0, max(max_degree, 10000),
            size=(total_walks, walk_length),
            dtype=np.int32
        )
        
        # Pre-allocate walks array
        walks_arr = np.full((total_walks, walk_length), -1, dtype=np.int32)
        walk_lengths = np.zeros(total_walks, dtype=np.int32)
        
        for walk_idx in tqdm(range(total_walks), desc="  Metapath walks"):
            # Quyết định kịch bản
            use_behavioral = np.random.random() < behavioral_ratio
            
            if use_behavioral and n_behavioral > 0:
                # --- Kịch bản 1: Behavioral (P → P → P ...) ---
                start_node = int(nodes_with_edges[walk_idx % n_behavioral])
            else:
                # --- Kịch bản 2: Semantic (P → A → P → A → P ...) ---
                start_node = int(nodes_with_aisle[walk_idx % n_semantic])
            
            cur = start_node
            walks_arr[walk_idx, 0] = cur
            wlen = 1
            
            for step in range(1, walk_length):
                if use_behavioral:
                    # Behavioral: đi theo CO_OCCUR edge
                    s = int(indptr[cur])
                    e = int(indptr[cur + 1])
                    n_nb = e - s
                    if n_nb == 0:
                        # Hết đường → reset về start hoặc chuyển semantic
                        if n_semantic > 0:
                            cur = int(nodes_with_aisle[walk_idx % n_semantic])
                        else:
                            break
                    else:
                        nb_idx = int(rand_matrix[walk_idx, step]) % n_nb
                        cur = int(neighbors[s + nb_idx])
                else:
                    # Semantic: P → A → P
                    # Bước 1: từ P đi lên Aisle
                    aisle_id = self.product_to_aisle.get(cur)
                    if aisle_id is None:
                        # Fallback về behavioral
                        s = int(indptr[cur])
                        e = int(indptr[cur + 1])
                        n_nb = e - s
                        if n_nb > 0:
                            nb_idx = int(rand_matrix[walk_idx, step]) % n_nb
                            cur = int(neighbors[s + nb_idx])
                        else:
                            break
                    else:
                        # Bước 2: từ Aisle chọn sản phẩm khác
                        products_in_aisle = self.aisle_to_products.get(aisle_id, [])
                        # Lọc bỏ chính nó
                        candidates = [p for p in products_in_aisle if p != cur]
                        if candidates:
                            cur = candidates[int(rand_matrix[walk_idx, step]) % len(candidates)]
                        else:
                            # Fallback về behavioral
                            s = int(indptr[cur])
                            e = int(indptr[cur + 1])
                            n_nb = e - s
                            if n_nb > 0:
                                nb_idx = int(rand_matrix[walk_idx, step]) % n_nb
                                cur = int(neighbors[s + nb_idx])
                            else:
                                break
                
                walks_arr[walk_idx, step] = cur
                wlen = step + 1
            
            walk_lengths[walk_idx] = wlen
        
        walks = walks_arr
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
        
        # --- Train Word2Vec (Skip-gram + Negative Sampling) ---
        print(f"  Đang train Word2Vec (dim={self.params['embedding_dim']})...")
        from gensim.models import Word2Vec
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.params['embedding_dim'],
            window=self.params['window'],
            min_count=1,    # giữ tất cả product nodes
            negative=self.params['negative'],
            epochs=self.params['epochs'],
            workers=self.params['workers'],
            sg=1,
            seed=RANDOM_SEED,
        )
        
        # Extract embeddings
        self.embeddings = np.zeros((n_products, self.params['embedding_dim']))
        for node in range(n_products):
            pid = self.idx_to_product_id[node]
            pid_str = str(pid)
            if pid_str in self.model.wv:
                self.embeddings[node] = self.model.wv[pid_str]
        
        # Cache embedding norms
        self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        self._embedding_norms[self._embedding_norms == 0] = 1e-9
        
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print("Metapath2Vec: Fit hoàn tất.")
    
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
            top_k = MW_TOP_K
        
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
        Lưu model ra file (embeddings, metadata, Word2Vec model, IKG mappings).
        
        Args:
            path: đường dẫn thư mục đầu ra
        """
        os.makedirs(path, exist_ok=True)
        
        # Lưu embeddings
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        
        # Lưu graph + IKG mappings
        graph_serializable = {
            str(k): [(int(n), int(w)) for n, w in v]
            for k, v in self.graph.items()
        }
        metadata = {
            'params': self.params,
            'graph': graph_serializable,
            'product_id_to_idx': {str(k): int(v) for k, v in self.product_id_to_idx.items()},
            'idx_to_product_id': {str(k): int(v) for k, v in self.idx_to_product_id.items()},
            'product_to_aisle': {str(k): int(v) for k, v in self.product_to_aisle.items()},
            'aisle_to_products': {
                str(k): [int(p) for p in v]
                for k, v in self.aisle_to_products.items()
            },
        }
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        # Lưu Word2Vec model
        if self.model is not None:
            self.model.save(os.path.join(path, "word2vec.model"))
        
        print(f"Metapath2Vec: Đã lưu tại {path}")
    
    def load(self, path: str):
        """
        Load model từ file (embeddings, metadata, Word2Vec model, IKG mappings).
        
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
        
        # Load IKG mappings
        self.product_to_aisle = {int(k): int(v) for k, v in metadata.get('product_to_aisle', {}).items()}
        self.aisle_to_products = {
            int(k): [int(p) for p in v]
            for k, v in metadata.get('aisle_to_products', {}).items()
        }
        
        # Load Word2Vec model
        model_path = os.path.join(path, "word2vec.model")
        if os.path.exists(model_path):
            from gensim.models import Word2Vec
            self.model = Word2Vec.load(model_path)
        
        self.graph_csr = None
        
        print(f"Metapath2Vec: Đã load từ {path}")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  IKG: {len(self.graph)} nodes có CO_OCCUR, "
              f"{len(self.aisle_to_products)} aisles")