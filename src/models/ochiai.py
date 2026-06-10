"""
Ochiai + Confidence Score.
Pairwise co-occurrence model với score có hướng.
"""
import os
import json
import math
from collections import defaultdict
import numpy as np
from scipy import sparse
from tqdm import tqdm
import pandas as pd

from src.config import OCHIAI_MIN_SUPPORT, OCHIAI_TOP_K, RANDOM_SEED


class OchiaiModel:
    """
    Ochiai + Confidence Score.
    
    score(A → B) = ochiai(A,B) * conf(A→B) * log1p(cnt(A,B))
    
    - ochiai = cnt / sqrt(count(A) * count(B))  — đối xứng, normalize
    - conf_{A→B} = cnt / count(A)               — bất đối xứng, có hướng
    - log_ab = log1p(cnt)                       — frequency bonus
    """
    
    def __init__(self, min_support: int = None):
        self.min_support = min_support if min_support is not None else OCHIAI_MIN_SUPPORT
        self.cooc_matrix = None          # CSR (n_products, n_products) — co-occurrence counts
        self.product_counts = None       # array (n_products,) — count(A)
        self.n_products = 0
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
    
    def fit(self, order_products: pd.DataFrame, products_df: pd.DataFrame):
        """
        Xây co-occurrence matrix từ order_products.
        
        Args:
            order_products: DataFrame [order_id, product_id, ...]
            products_df: DataFrame [product_id, ...]
        """
        print("OchiaiModel: Bắt đầu fit...")
        
        # Mapping product_id → idx
        all_product_ids = sorted(products_df['product_id'].unique())
        self.n_products = len(all_product_ids)
        self.product_id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        self.idx_to_product_id = {i: pid for pid, i in self.product_id_to_idx.items()}
        
        print(f"  Số sản phẩm: {self.n_products}")
        print(f"  Đang đếm co-occurrence pairs...")
        
        # Đếm cặp (i,j) bằng dictionary
        pair_counts = defaultdict(int)
        product_counts = defaultdict(int)
        
        # Duyệt theo order_id
        # Nhóm order_products theo order_id
        grouped = order_products.groupby('order_id')['product_id']
        
        for order_id, group in tqdm(grouped, desc="  Duyệt orders", unit="order"):
            items = list(group)
            product_ids_in_order = [
                self.product_id_to_idx[pid] for pid in items
                if pid in self.product_id_to_idx
            ]
            
            # Đếm product counts
            for idx in product_ids_in_order:
                product_counts[idx] += 1
            
            # Đếm pair counts (i < j)
            n = len(product_ids_in_order)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = product_ids_in_order[i], product_ids_in_order[j]
                    if a != b:
                        pair_counts[(a, b)] += 1
        
        print(f"  Tổng số pairs (raw): {len(pair_counts)}")
        
        # Lọc min_support
        if self.min_support > 0:
            pair_counts = {
                pair: cnt for pair, cnt in pair_counts.items()
                if cnt >= self.min_support
            }
            print(f"  Sau min_support={self.min_support}: {len(pair_counts)} pairs")
        
        # Chuyển sang CSR matrix
        print("  Xây CSR matrix...")
        rows, cols, data = [], [], []
        for (a, b), cnt in pair_counts.items():
            rows.append(a)
            cols.append(b)
            data.append(cnt)
            # Ma trận đối xứng
            rows.append(b)
            cols.append(a)
            data.append(cnt)
        
        self.cooc_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_products, self.n_products),
            dtype=np.int32
        )
        
        # Product counts array
        self.product_counts = np.zeros(self.n_products, dtype=np.int32)
        for idx, cnt in product_counts.items():
            self.product_counts[idx] = cnt
        
        print(f"  CSR matrix shape: {self.cooc_matrix.shape}")
        print(f"  Non-zero entries: {self.cooc_matrix.nnz}")
        
        # Tính total orders
        self.total_orders = order_products['order_id'].nunique()
        print(f"  Total orders: {self.total_orders}")
        
        print("OchiaiModel: Fit hoàn tất.")
    
    def _compute_scores(self, product_idx: int):
        """
        Tính score cho product_idx với tất cả các product khác.
        
        score(i→j) = ochiai(i,j) * conf(i→j) * log1p(cnt(i,j))
        
        Args:
            product_idx: int, index của sản phẩm đầu vào
        
        Returns:
            numpy array (n_products,) — score từ product_idx đến mọi product
        """
        cnt_i = self.product_counts[product_idx]
        if cnt_i == 0:
            return np.zeros(self.n_products)
        
        # Lấy row của cooc_matrix (các product B có co-occurrence với A)
        row = self.cooc_matrix[product_idx].toarray().flatten()
        
        # Ochiai = cnt / sqrt(count(A) * count(B))
        ochiai = np.zeros(self.n_products)
        nonzero_indices = np.where(row > 0)[0]
        
        # Vectorized: tính ochiai cho tất cả non-zero entries cùng lúc
        cnts = row[nonzero_indices]
        counts_j = self.product_counts[nonzero_indices]
        mask = counts_j > 0
        if mask.any():
            valid_indices = nonzero_indices[mask]
            ochiai[valid_indices] = cnts[mask] / np.sqrt(cnt_i * counts_j[mask])
        
        # Confidence: conf(A→B) = cnt / count(A)
        conf = row / cnt_i
        
        # Log frequency: log1p(cnt)
        log_freq = np.log1p(row)
        
        # Score cuối
        scores = ochiai * conf * log_freq
        return scores
    
    def recommend(self, product_id: int, top_k: int = None):
        """
        Trả về top-K product có score cao nhất cho product_id đầu vào.
        
        Args:
            product_id: int
            top_k: int, số lượng gợi ý (default: OCHIAI_TOP_K)
        
        Returns:
            list (product_id, score) sorted giảm dần
        """
        if top_k is None:
            top_k = OCHIAI_TOP_K
        
        if product_id not in self.product_id_to_idx:
            return []
        
        idx = self.product_id_to_idx[product_id]
        scores = self._compute_scores(idx)
        
        # Bỏ qua chính nó
        scores[idx] = -1
        
        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        result = [
            (self.idx_to_product_id[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]
        
        return result
    
    def save(self, path: str):
        """
        Lưu model ra file.
        
        Args:
            path: đường dẫn thư mục (vd: models/ochiai/)
        """
        os.makedirs(path, exist_ok=True)
        
        # Lưu CSR matrix
        sparse.save_npz(os.path.join(path, "cooc_matrix.npz"), self.cooc_matrix)
        
        # Lưu metadata — convert int64 → int để JSON serialize được
        metadata = {
            'min_support': int(self.min_support),
            'n_products': int(self.n_products),
            'product_id_to_idx': {int(k): int(v) for k, v in self.product_id_to_idx.items()},
            'idx_to_product_id': {str(int(k)): int(v) for k, v in self.idx_to_product_id.items()},
            'product_counts': [int(c) for c in self.product_counts],
            'total_orders': int(self.total_orders),
        }
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        # Lưu product counts
        np.savetxt(os.path.join(path, "product_counts.csv"),
                   np.column_stack([list(self.idx_to_product_id.values()),
                                    self.product_counts]),
                   delimiter=',', header='product_id,count', comments='',
                   fmt='%d')
        
        print(f"OchiaiModel: Đã lưu tại {path}")
    
    def load(self, path: str):
        """
        Load model từ file.
        
        Args:
            path: đường dẫn thư mục (vd: models/ochiai/)
        """
        self.cooc_matrix = sparse.load_npz(os.path.join(path, "cooc_matrix.npz"))
        
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        self.min_support = metadata['min_support']
        self.n_products = metadata['n_products']
        self.product_id_to_idx = metadata['product_id_to_idx']
        self.product_id_to_idx = {int(k): int(v) for k, v in self.product_id_to_idx.items()}
        self.idx_to_product_id = {int(k): int(v) for k, v in metadata['idx_to_product_id'].items()}
        self.product_counts = np.array(metadata['product_counts'], dtype=np.int32)
        self.total_orders = int(metadata['total_orders'])
        
        print(f"OchiaiModel: Đã load từ {path}")
        print(f"  Số sản phẩm: {self.n_products}")
        print(f"  Non-zero entries: {self.cooc_matrix.nnz}")