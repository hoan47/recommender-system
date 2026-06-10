"""
Content-Based Diversity Filter.
Bộ lọc hậu xử lý — loại bỏ substitute (sản phẩm thay thế/quá giống)
khỏi kết quả gợi ý của các model co-occurrence.
"""
import pandas as pd
import numpy as np
from scipy import sparse

from src.config import CB_THRESHOLD, CB_N_GRAM_RANGE, CB_MAX_FEATURES, PRODUCT_VECTORS_FILE
from src.features.vectorizer import build_product_vectors, cb_similarity


class CBFilter:
    """
    Content-Based Diversity Filter.
    
    Chỉ tính cosine similarity on-demand cho các cặp được đề xuất,
    không pre-compute full matrix 49K x 49K.
    """
    
    def __init__(self, threshold: float = None):
        """
        Args:
            threshold: cosine similarity >= threshold → substitute → loại bỏ
        """
        self.threshold = threshold if threshold is not None else CB_THRESHOLD
        self.product_vectors = None  # sparse.csr_matrix (n_products, D)
        self.product_id_to_idx = {}  # mapping product_id → row index in matrix
    
    def fit(self, products_df, ngram_range=None, max_features=None):
        """
        Pre-compute product vectors 1 lần.
        Lưu product_vectors và mapping product_id → idx.
        
        Args:
            products_df: DataFrame [product_id, product_name, aisle_id, department_id, ...]
            ngram_range: tuple (min_n, max_n) cho TF-IDF
            max_features: int, max features cho TF-IDF
        """
        if ngram_range is None:
            ngram_range = CB_N_GRAM_RANGE
        if max_features is None:
            max_features = CB_MAX_FEATURES
        
        print("CBFilter: Đang vector hóa sản phẩm...")
        self.product_vectors, _ = build_product_vectors(
            products_df, ngram_range=ngram_range, max_features=max_features
        )
        
        # Mapping product_id → idx (dựa trên thứ tự trong products_df)
        for i, pid in enumerate(products_df['product_id']):
            self.product_id_to_idx[pid] = i
        
        print(f"CBFilter: Đã vector hóa {len(self.product_id_to_idx)} sản phẩm.")
    
    def filter(self, product_a_id: int, candidates):
        """
        Lọc substitute khỏi danh sách candidate.
        
        Args:
            product_a_id: product_id đầu vào
            candidates: list (product_id, score) đã sort giảm dần theo score,
                       hoặc list product_id đơn thuần
        
        Returns:
            list (product_id, score) đã loại bỏ substitute,
            hoặc list product_id nếu đầu vào là list đơn thuần
        """
        if product_a_id not in self.product_id_to_idx:
            # Cold-start: không có vector → giữ nguyên candidates
            return candidates
        
        idx_a = self.product_id_to_idx[product_a_id]
        
        # Xác định định dạng đầu vào
        is_tuple_list = candidates and isinstance(candidates[0], (list, tuple))
        
        if is_tuple_list:
            candidate_ids = [c[0] for c in candidates]
        else:
            candidate_ids = list(candidates)
        
        # Lọc các candidate có vector
        valid_indices = []
        valid_candidates = []
        for cid in candidate_ids:
            if cid in self.product_id_to_idx:
                valid_indices.append(self.product_id_to_idx[cid])
                valid_candidates.append(cid)
        
        if not valid_indices:
            return candidates
        
        # Tính similarity on-demand
        similarities = cb_similarity(self.product_vectors, idx_a, valid_indices)
        
        # Loại bỏ substitute (similarity >= threshold)
        mask = similarities < self.threshold
        
        if is_tuple_list:
            # Tra cứu similarity nhanh bằng dict, giữ nguyên thứ tự
            valid_set = set(self.product_id_to_idx.keys())
            sim_map = dict(zip(valid_candidates, similarities))
            
            result = []
            for cid, score in candidates:
                if cid not in valid_set:
                    result.append((cid, score))      # cold-start: giữ lại
                elif sim_map.get(cid, 0) < self.threshold:
                    result.append((cid, score))      # complementary: giữ lại
                # else: substitute (similarity >= threshold) → bỏ
            return result
        else:
            # Trả về list product_id
            filtered = [
                candidates[i] for i, cid in enumerate(candidate_ids)
                if cid in valid_candidates
            ]
            result = [c for c, keep in zip(filtered, mask) if keep]
            # Thêm lại các candidate không có vector (cold-start)
            cold_start = [
                cid for cid in candidate_ids
                if cid not in self.product_id_to_idx
            ]
            # Giữ đúng thứ tự ban đầu
            ordered_result = []
            for cid in candidate_ids:
                if cid in cold_start:
                    ordered_result.append(cid)
                elif cid in set(result):
                    ordered_result.append(cid)
            return ordered_result
    
    def filter_df(self, product_a_id: int, candidate_df, score_col: str = "score"):
        """
        Version trả về DataFrame thay vì list.
        
        Args:
            product_a_id: product_id đầu vào
            candidate_df: DataFrame [product_id, score, ...]
            score_col: tên cột chứa score
        
        Returns:
            DataFrame đã loại bỏ substitute
        """
        if product_a_id not in self.product_id_to_idx:
            return candidate_df
        
        idx_a = self.product_id_to_idx[product_a_id]
        
        # Lọc các candidate có vector
        valid_df = candidate_df[
            candidate_df['product_id'].isin(self.product_id_to_idx)
        ]
        if valid_df.empty:
            return candidate_df
        
        valid_indices = [
            self.product_id_to_idx[pid]
            for pid in valid_df['product_id']
        ]
        
        # Tính similarity
        similarities = cb_similarity(self.product_vectors, idx_a, valid_indices)
        
        # Giữ lại complementary (similarity < threshold)
        mask = np.array(similarities) < self.threshold
        filtered_df = valid_df[mask].copy()
        
        # Thêm lại cold-start candidates
        cold_start_df = candidate_df[
            ~candidate_df['product_id'].isin(self.product_id_to_idx)
        ]
        
        return pd.concat([filtered_df, cold_start_df], ignore_index=True)