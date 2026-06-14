"""
Co-occurrence Ensemble + CB Filter.
Kết hợp Ochiai, Item2Vec, DeepWalk bằng weighted score, sau đó lọc substitute bằng CB.
"""
import numpy as np
import pandas as pd

from src.config import (
    ENS_ALPHA, ENS_BETA, ENS_GAMMA,
    ENS_TOP_K, ENS_FINAL_K
)
from src.models.cb_filter import CBFilter


class EnsembleModel:
    """
    Co-occurrence Ensemble + CB Filter.
    
    final_score(A → B) = α × Ochiai_score(A, B)
                        + β × Item2Vec_sim(A, B)
                        + γ × DeepWalk_sim(A, B)
    """
    
    def __init__(self, alpha=None, beta=None, gamma=None,
                 top_k=None, final_k=None):
        self.alpha = alpha if alpha is not None else ENS_ALPHA
        self.beta = beta if beta is not None else ENS_BETA
        self.gamma = gamma if gamma is not None else ENS_GAMMA
        self.top_k = top_k if top_k is not None else ENS_TOP_K
        self.final_k = final_k if final_k is not None else ENS_FINAL_K
        
        self.ochiai = None
        self.item2vec = None
        self.deepwalk = None
        self.cb_filter = None
    
    def fit(self, ochiai, item2vec, deepwalk, cb_filter: CBFilter = None):
        """
        Gán các model đã train.
        
        Args:
            ochiai: OchiaiModel instance
            item2vec: Item2VecModel instance
            deepwalk: DeepWalkModel instance
            cb_filter: CBFilter instance (optional)
        """
        self.ochiai = ochiai
        self.item2vec = item2vec
        self.deepwalk = deepwalk
        self.cb_filter = cb_filter
        
        print(f"Ensemble: Đã gán models (α={self.alpha}, β={self.beta}, γ={self.gamma})")
    
    def _normalize(self, scores):
        """
        Min-max normalize scores về [0, 1].
        
        Args:
            scores: list of float
        
        Returns:
            list of float đã normalize
        """
        if not scores:
            return []
        
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        
        return [(s - min_s) / (max_s - min_s + 1e-9) for s in scores]
    
    def _build_score_dict(self, recommendations):
        """
        Convert list (id, score) thành dict {id: score}.
        
        Args:
            recommendations: list (product_id, score)
        
        Returns:
            dict: {product_id: score}
        """
        return {pid: score for pid, score in recommendations}
    
    def get_raw_candidates(self, product_id: int, top_k: int = None):
        """
        Lấy raw candidates từ union của 3 model (không CB filter, không giới hạn final_k).
        Useful cho evaluation/survey.

        Args:
            product_id: int
            top_k: số candidate lấy từ mỗi model (default: self.top_k)

        Returns:
            list (product_id, ensemble_score) — sorted descending, không giới hạn số lượng
        """
        if top_k is None:
            top_k = self.top_k

        # Lấy recommendations từ từng model
        ochiai_recs = self.ochiai.recommend(product_id, top_k=top_k) if self.ochiai else []
        i2v_recs = self.item2vec.recommend(product_id, top_k=top_k) if self.item2vec else []
        dw_recs = self.deepwalk.recommend(product_id, top_k=top_k) if self.deepwalk else []

        # Union candidates
        candidate_ids = set()
        for pid, _ in ochiai_recs:
            candidate_ids.add(pid)
        for pid, _ in i2v_recs:
            candidate_ids.add(pid)
        for pid, _ in dw_recs:
            candidate_ids.add(pid)

        if not candidate_ids:
            return []

        # Score dicts
        ochiai_dict = self._build_score_dict(ochiai_recs)
        i2v_dict = self._build_score_dict(i2v_recs)
        dw_dict = self._build_score_dict(dw_recs)

        # Tính weighted score
        candidate_list = list(candidate_ids)
        ochiai_scores = [ochiai_dict.get(pid, 0) for pid in candidate_list]
        i2v_scores = [i2v_dict.get(pid, 0) for pid in candidate_list]
        dw_scores = [dw_dict.get(pid, 0) for pid in candidate_list]

        ochiai_norm = self._normalize(ochiai_scores)
        i2v_norm = self._normalize(i2v_scores)
        dw_norm = self._normalize(dw_scores)

        final_scores = [
            self.alpha * o + self.beta * i + self.gamma * d
            for o, i, d in zip(ochiai_norm, i2v_norm, dw_norm)
        ]

        # Sort descending
        sorted_indices = np.argsort(final_scores)[::-1]
        result = [
            (candidate_list[i], final_scores[i])
            for i in sorted_indices
            if final_scores[i] > 0
        ]

        return result

    def recommend(self, product_id: int, use_cb_filter: bool = True):
        """
        Ensemble recommendation.
        
        1. Lấy top-K candidate từ mỗi model (O, I, D)
        2. Union các candidate lại
        3. Tính weighted score cho mỗi candidate
        4. Sort descending
        5. (Optional) CB Filter loại substitute
        6. Trả về final_k gợi ý
        
        Args:
            product_id: int, sản phẩm đầu vào
            use_cb_filter: bool, có dùng CB Filter không
        
        Returns:
            list (product_id, final_score)
        """
        # Lấy raw candidates
        candidates_sorted = self.get_raw_candidates(product_id, top_k=self.top_k)
        
        if not candidates_sorted:
            return []
        
        # --- Bước 5: CB Filter (optional) ---
        if use_cb_filter and self.cb_filter is not None:
            candidates_sorted = self.cb_filter.filter(product_id, candidates_sorted)
        
        # --- Bước 6: Lấy final_k ---
        result = candidates_sorted[:self.final_k]
        
        return result
    
    def recommend_all(self, product_ids, use_cb_filter: bool = True):
        """
        Batch recommend cho nhiều product.
        
        Args:
            product_ids: list[int]
            use_cb_filter: bool
        
        Returns:
            dict: {product_id: list (product_id, score)}
        """
        return {
            pid: self.recommend(pid, use_cb_filter=use_cb_filter)
            for pid in product_ids
        }
    
    def compare_without_filter(self, product_id: int):
        """
        So sánh kết quả có và không có CB Filter.
        
        Args:
            product_id: int
        
        Returns:
            dict: {'with_filter': [...], 'without_filter': [...]}
        """
        return {
            'with_filter': self.recommend(product_id, use_cb_filter=True),
            'without_filter': self.recommend(product_id, use_cb_filter=False),
        }