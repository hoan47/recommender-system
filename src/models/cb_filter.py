"""
Content-Based Diversity Filter.
Bộ lọc hậu xử lý — loại bỏ substitute (sản phẩm thay thế/quá giống)
khỏi kết quả gợi ý của các model co-occurrence.

Cải tiến: Ensemble Count Vectorizer + TF-IDF để kết hợp
ưu điểm của cả count (tuyến tính theo từ trùng) và TF-IDF (phân hóa từ khóa).
"""
import pandas as pd
import numpy as np
from scipy import sparse

from src.config import (
    CB_N_GRAM_RANGE, CB_MAX_FEATURES,
    CB_COUNT_N_GRAM_RANGE, CB_COUNT_MAX_FEATURES,
    CB_ALPHA,
    CB_NAME_WEIGHT, CB_AISLE_WEIGHT, CB_DEPT_WEIGHT,
    CB_AISLE_N_GRAM_RANGE, CB_AISLE_MAX_FEATURES,
    CB_DEPT_N_GRAM_RANGE, CB_DEPT_MAX_FEATURES,
)
from src.features.vectorizer import (
    build_product_vectors, build_count_vectors,
    build_multi_field_vectors,
    cb_ensemble_similarity,
)


class CBFilter:
    """
    Content-Based Diversity Filter.

    Sử dụng ensemble Count + TF-IDF để tính similarity giữa sản phẩm.
    Chỉ tính on-demand cho các cặp được đề xuất,
    """

    def __init__(self, ngram_range=None, max_features: int = None,
                 count_ngram_range=None, count_max_features: int = None,
                 alpha: float = None, metric: str = 'overlap'):
        """
        Args:
            ngram_range: tuple (min_n, max_n) cho TF-IDF (default: CB_N_GRAM_RANGE)
            max_features: int, max features cho TF-IDF (default: CB_MAX_FEATURES)
            count_ngram_range: tuple (min_n, max_n) cho Count Vectorizer
                               (default: CB_COUNT_N_GRAM_RANGE)
            count_max_features: int, max features cho Count Vectorizer
                               (default: CB_COUNT_MAX_FEATURES)
            alpha: float, trọng số Count Vectorizer (TF-IDF weight = 1-alpha)
                   (default: CB_ALPHA)
            metric: str, 'overlap' cho nhánh Count (Overlap Score)
        """
        self.ngram_range = ngram_range if ngram_range is not None else CB_N_GRAM_RANGE
        self.max_features = max_features if max_features is not None else CB_MAX_FEATURES
        self.count_ngram_range = (count_ngram_range if count_ngram_range is not None
                                  else CB_COUNT_N_GRAM_RANGE)
        self.count_max_features = (count_max_features if count_max_features is not None
                                   else CB_COUNT_MAX_FEATURES)
        self.alpha = alpha if alpha is not None else CB_ALPHA
        self.metric = metric if metric is not None else 'overlap'

        self.product_vectors_tfidf = None   # sparse.csr_matrix (n_products, D_tfidf)
        self.product_vectors_count = None   # sparse.csr_matrix (n_products, D_count)
        self.product_id_to_idx = {}          # mapping product_id → row index in matrix

    @property
    def product_vectors(self):
        """Tương thích ngược: trỏ đến TF-IDF vectors."""
        return self.product_vectors_tfidf

    @product_vectors.setter
    def product_vectors(self, value):
        """Tương thích ngược cho load từ file (ensemble.py)."""
        self.product_vectors_tfidf = value

    def fit(self, products_df, ngram_range=None, max_features=None):
        """
        Pre-compute product vectors 1 lần — vector hóa sản phẩm bằng cả
        Count Vectorizer và TF-IDF multi-field (name + aisle + department).

        Args:
            products_df: DataFrame [product_id, product_name, aisle, department, ...]
            (product_name, aisle, department đã là tiếng Việt sau load_products)
            ngram_range: tuple (min_n, max_n) cho TF-IDF (default: self.ngram_range)
            max_features: int, max features cho TF-IDF (default: self.max_features)
        """
        if ngram_range is None:
            ngram_range = self.ngram_range
        if max_features is None:
            max_features = self.max_features

        print("CBFilter: Đang vector hóa sản phẩm (tiếng Việt, multi-field)...")
        # Lấy text_data từ các cột tiếng Việt
        text_name = products_df['product_name'].fillna('').tolist()
        text_aisle = products_df['aisle'].fillna('').tolist()
        text_dept = products_df['department'].fillna('').tolist()

        # 1. TF-IDF multi-field: name + aisle + department (có trọng số)
        print("  [1/2] TF-IDF multi-field (name + aisle + department):")
        fields_dict = {
            'name': {
                'texts': text_name,
                'weight': CB_NAME_WEIGHT,
                'ngram_range': ngram_range,
                'max_features': max_features,
            },
            'aisle': {
                'texts': text_aisle,
                'weight': CB_AISLE_WEIGHT,
                'ngram_range': CB_AISLE_N_GRAM_RANGE,
                'max_features': CB_AISLE_MAX_FEATURES,
            },
            'dept': {
                'texts': text_dept,
                'weight': CB_DEPT_WEIGHT,
                'ngram_range': CB_DEPT_N_GRAM_RANGE,
                'max_features': CB_DEPT_MAX_FEATURES,
            },
        }
        self.product_vectors_tfidf, self._vectorizers = build_multi_field_vectors(fields_dict)

        # 2. Count vectors (L2-normalized) — chỉ trên product_name (giữ nguyên)
        print("  [2/2] Count vectors (L2-normalized) — chỉ trên product_name:")
        self.product_vectors_count, _ = build_count_vectors(
            text_name,
            ngram_range=self.count_ngram_range,
            max_features=self.count_max_features,
        )

        # Mapping product_id → idx (dựa trên thứ tự trong products_df)
        for i, pid in enumerate(products_df['product_id']):
            self.product_id_to_idx[pid] = i

        print(f"CBFilter: Đã vector hóa {len(self.product_id_to_idx)} sản phẩm.")
        print(f"  TF-IDF multi-field shape:  {self.product_vectors_tfidf.shape}")
        print(f"  Count shape:               {self.product_vectors_count.shape}")
        print(f"  Alpha (Count)              = {self.alpha:.2f}")
        print(f"  Trọng số: name={CB_NAME_WEIGHT}, aisle={CB_AISLE_WEIGHT}, dept={CB_DEPT_WEIGHT}")

    def filter(self, product_a_id: int, candidates, threshold: float):
        """
        Lọc substitute khỏi danh sách candidate — giữ lại complementary.

        Args:
            product_a_id: product_id đầu vào
            candidates: list (product_id, score) đã sort giảm dần theo score,
                       hoặc list product_id đơn thuần
            threshold: cosine similarity >= threshold → substitute → loại bỏ.
                       Chỉ dùng khi hybrid ensemble, CB tự nó không biết ngưỡng.

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

        # Tính ensemble similarity on-demand
        similarities = cb_ensemble_similarity(
            self.product_vectors_tfidf, self.product_vectors_count,
            idx_a, valid_indices, alpha=self.alpha,
        )

        # Loại bỏ:
        # - similarity >= threshold → substitute → bỏ
        # Giữ: similarity < threshold (gồm cả similarity == 0 → complementary mạnh nhất)
        mask = similarities < threshold

        if is_tuple_list:
            # Tra cứu similarity nhanh bằng dict, giữ nguyên thứ tự
            valid_set = set(self.product_id_to_idx.keys())
            sim_map = dict(zip(valid_candidates, similarities))

            result = []
            for cid, score in candidates:
                if cid not in valid_set:
                    result.append((cid, score))      # cold-start: giữ lại
                else:
                    sim = sim_map.get(cid, 0)
                    if sim < threshold:
                        result.append((cid, score))  # complementary: giữ lại
                # else: substitute (sim >= threshold) → bỏ
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

    def filter_df(self, product_a_id: int, candidate_df, threshold: float):
        """
        Version trả về DataFrame thay vì list — lọc substitute.

        Args:
            product_a_id: product_id đầu vào
            candidate_df: DataFrame [product_id, score, ...]
            threshold: cosine similarity >= threshold → substitute → loại bỏ

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

        # Tính ensemble similarity
        similarities = cb_ensemble_similarity(
            self.product_vectors_tfidf, self.product_vectors_count,
            idx_a, valid_indices, alpha=self.alpha,
        )

        # Chỉ giữ lại: similarity < threshold (gồm cả similarity == 0)
        mask = np.array(similarities) < threshold
        filtered_df = valid_df[mask].copy()

        # Thêm lại cold-start candidates
        cold_start_df = candidate_df[
            ~candidate_df['product_id'].isin(self.product_id_to_idx)
        ]

        return pd.concat([filtered_df, cold_start_df], ignore_index=True)