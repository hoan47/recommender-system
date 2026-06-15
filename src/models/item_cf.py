"""
Item-Based Collaborative Filtering (Item-CF) — Ochiai + Confidence Score.
Trước đây gọi là OchiaiModel, đổi tên cho đúng bản chất thuật toán:
Ochiai coefficient = Cosine similarity trên binary vector (mỗi sản phẩm
là binary vector với chiều là các order) → Item-Based CF.

Pairwise co-occurrence model với score có hướng.

Cải tiến Numba: dùng count_pairs_numba() để đếm co-occurrence pairs
nhanh hơn defaultdict + combinations.
"""
import os
import json
import numpy as np
from scipy import sparse
from tqdm import tqdm
import pandas as pd

from src.config import ITEMCF_MIN_SUPPORT, ITEMCF_TOP_K
from src.utils._numba_ops import count_pairs_numba


class ItemCFModel:
    """
    Item-Based Collaborative Filtering (Item-CF) — Ochiai + Confidence Score.
    
    score(A → B) = ochiai(A,B) * conf(A→B) * log1p(cnt(A,B))
    
    - ochiai = cnt / sqrt(count(A) * count(B))  — đối xứng, normalize
      (tương đương Cosine similarity trên binary vector)
    - conf_{A→B} = cnt / count(A)               — bất đối xứng, có hướng
    - log_ab = log1p(cnt)                       — frequency bonus
    """
    
    def __init__(self, min_support: int = None, top_k: int = None):
        self.min_support = min_support if min_support is not None else ITEMCF_MIN_SUPPORT
        self.top_k = top_k if top_k is not None else ITEMCF_TOP_K
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
        print("ItemCFModel: Bắt đầu fit...")
        
        # Mapping product_id → idx
        all_product_ids = sorted(products_df['product_id'].unique())
        self.n_products = len(all_product_ids)
        self.product_id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        self.idx_to_product_id = {i: pid for pid, i in self.product_id_to_idx.items()}
        
        print(f"  Số sản phẩm: {self.n_products}")
        print(f"  Đang xây order_indices (CSR-like) cho Numba counting...")
        
        # Xây CSR-like representation của orders để đưa vào Numba
        grouped = order_products.groupby('order_id')['product_id']
        
        # Đếm tổng số items để pre-allocate
        total_items = 0
        order_lengths = []
        for order_id, group in tqdm(grouped, desc="  Scan orders", unit="order"):
            items = [self.product_id_to_idx.get(pid, -1) for pid in group]
            items = [x for x in items if x >= 0]
            if items:
                total_items += len(items)
                order_lengths.append(len(items))
        
        # Build order_indices và order_ptr
        order_indices = np.zeros(total_items, dtype=np.int32)
        order_ptr = np.zeros(len(order_lengths) + 1, dtype=np.int32)
        
        pos = 0
        for o_idx, length in enumerate(order_lengths):
            order_ptr[o_idx] = pos
            pos += length
        order_ptr[-1] = total_items
        
        # Fill lại order_indices
        pos = 0
        for order_id, group in tqdm(grouped, desc="  Fill indices", unit="order"):
            items = [self.product_id_to_idx.get(pid, -1) for pid in group]
            items = [x for x in items if x >= 0]
            n = len(items)
            if n > 0:
                order_indices[pos:pos + n] = items
                pos += n
        
        print(f"  Tổng items: {total_items}, tổng orders: {len(order_lengths)}")
        print(f"  Đang đếm co-occurrence pairs bằng Numba...")
        
        # Numba counting
        rows, cols, counts = count_pairs_numba(
            order_indices, order_ptr, self.n_products
        )
        
        print(f"  Tổng số pairs (unique, raw): {len(rows)}")
        
        # Lọc min_support
        if self.min_support > 0:
            mask = counts >= self.min_support
            rows = rows[mask]
            cols = cols[mask]
            counts = counts[mask]
            print(f"  Sau min_support={self.min_support}: {len(rows)} pairs")
        
        # Xây CSR matrix
        print("  Xây CSR matrix...")
        # Ma trận đối xứng: thêm cả (b, a)
        rows_all = np.concatenate([rows, cols])
        cols_all = np.concatenate([cols, rows])
        data_all = np.concatenate([counts, counts])
        
        self.cooc_matrix = sparse.csr_matrix(
            (data_all, (rows_all, cols_all)),
            shape=(self.n_products, self.n_products),
            dtype=np.int32
        )
        
        # Product counts array — đếm từ order_indices
        print("  Đếm product counts...")
        self.product_counts = np.bincount(
            order_indices, minlength=self.n_products
        ).astype(np.int32)
        
        print(f"  CSR matrix shape: {self.cooc_matrix.shape}")
        print(f"  Non-zero entries: {self.cooc_matrix.nnz}")
        
        # Tính total orders
        self.total_orders = order_products['order_id'].nunique()
        print(f"  Total orders: {self.total_orders}")
        
        print("ItemCFModel: Fit hoàn tất.")
    
    def _compute_scores(self, product_idx: int):
        """
        Tính score cho product_idx với tất cả các product khác.

        score(i→j) = ochiai(i,j) * conf(i→j) * log1p(cnt(i,j))

        Args:
            product_idx: int — index của sản phẩm đầu vào

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
        Trả về top-K product có score cao nhất cho product_id đầu vào — gợi ý mua kèm.

        Args:
            product_id: int — ID sản phẩm đầu vào
            top_k: int — số lượng gợi ý (default: ITEMCF_TOP_K)

        Returns:
            list (product_id, score) — sorted giảm dần theo score
        """
        if top_k is None:
            top_k = self.top_k
        
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
        Lưu model ra file (cooc matrix + metadata + product counts).

        Args:
            path: đường dẫn thư mục đầu ra (vd: models/item_cf/)
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
        
        print(f"ItemCFModel: Đã lưu tại {path}")
    
    def load(self, path: str):
        """
        Load model từ file (cooc matrix + metadata + product counts).

        Args:
            path: đường dẫn thư mục đã lưu (vd: models/item_cf/)
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
        
        print(f"ItemCFModel: Đã load từ {path}")
        print(f"  Số sản phẩm: {self.n_products}")
        print(f"  Non-zero entries: {self.cooc_matrix.nnz}")