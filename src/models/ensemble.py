"""
Co-occurrence Ensemble + CB Filter.
Kết hợp Item-CF (Ochiai), Item2Vec, Metapath2Vec bằng weighted score, sau đó lọc substitute bằng CB.
"""
import json
import os
import numpy as np
import pandas as pd
import scipy.sparse

from src.config import (
    ENS_ALPHA, ENS_BETA, ENS_GAMMA,
    ENS_TOP_K, ENS_FINAL_K, MODEL_DIR, ENS_CB_THRESHOLD
)
from src.models.cb_filter import CBFilter


class EnsembleModel:
    """
    Co-occurrence Ensemble + CB Filter.
    
    final_score(A → B) = α × ItemCF_score(A, B)
                        + β × Item2Vec_sim(A, B)
                        + γ × Metapath2Vec_sim(A, B)
    
    Có thể save/load toàn bộ model ensemble (bao gồm các sub-models) bằng 1 lệnh.
    """
    
    def __init__(self, alpha=None, beta=None, gamma=None,
                 top_k=None, final_k=None):
        self.alpha = alpha if alpha is not None else ENS_ALPHA
        self.beta = beta if beta is not None else ENS_BETA
        self.gamma = gamma if gamma is not None else ENS_GAMMA
        self.top_k = top_k if top_k is not None else ENS_TOP_K
        self.final_k = final_k if final_k is not None else ENS_FINAL_K
        
        self.item_cf = None
        self.item2vec = None
        self.metapath2vec = None
        self.cb_filter = None
    
    def fit(self, item_cf, item2vec, metapath2vec, cb_filter: CBFilter = None):
        """
        Gán các model đã train.
        
        Args:
            item_cf: ItemCFModel instance (Item-Based Collaborative Filtering)
            item2vec: Item2VecModel instance
            metapath2vec: Metapath2VecModel instance
            cb_filter: CBFilter instance (optional)
        """
        self.item_cf = item_cf
        self.item2vec = item2vec
        self.metapath2vec = metapath2vec
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
        item_cf_recs = self.item_cf.recommend(product_id, top_k=top_k) if self.item_cf else []
        i2v_recs = self.item2vec.recommend(product_id, top_k=top_k) if self.item2vec else []
        mw_recs = self.metapath2vec.recommend(product_id, top_k=top_k) if self.metapath2vec else []

        # Union candidates
        candidate_ids = set()
        for pid, _ in item_cf_recs:
            candidate_ids.add(pid)
        for pid, _ in i2v_recs:
            candidate_ids.add(pid)
        for pid, _ in mw_recs:
            candidate_ids.add(pid)

        if not candidate_ids:
            return []

        # Score dicts
        item_cf_dict = self._build_score_dict(item_cf_recs)
        i2v_dict = self._build_score_dict(i2v_recs)
        mw_dict = self._build_score_dict(mw_recs)

        # Tính weighted score
        candidate_list = list(candidate_ids)
        item_cf_scores = [item_cf_dict.get(pid, 0) for pid in candidate_list]
        i2v_scores = [i2v_dict.get(pid, 0) for pid in candidate_list]
        mw_scores = [mw_dict.get(pid, 0) for pid in candidate_list]

        item_cf_norm = self._normalize(item_cf_scores)
        i2v_norm = self._normalize(i2v_scores)
        mw_norm = self._normalize(mw_scores)

        final_scores = [
            self.alpha * o + self.beta * i + self.gamma * m
            for o, i, m in zip(item_cf_norm, i2v_norm, mw_norm)
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
        
        1. Lấy top-K candidate từ mỗi model (Item-CF, Item2Vec, Metapath2Vec)
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
            candidates_sorted = self.cb_filter.filter(
                product_id, candidates_sorted, threshold=ENS_CB_THRESHOLD
            )
        
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
    
    def save(self, path: str = None):
        """
        Lưu ensemble model (config + cb_filter) vào thư mục.
        
        Chỉ lưu config và CB Filter (sub-models đã được lưu riêng ở steps 03-05, 05 là Metapath2Vec).
        
        Args:
            path: đường dẫn thư mục lưu (default: MODEL_DIR/ensemble)
        """
        if path is None:
            path = os.path.join(MODEL_DIR, "ensemble")
        os.makedirs(path, exist_ok=True)
        
        # Lưu config
        config = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'top_k': self.top_k,
            'final_k': self.final_k,
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Lưu CB Filter vectors (nếu có) — để load sau không cần đọc từ cb_filter/
        if self.cb_filter is not None and self.cb_filter.product_vectors_tfidf is not None:
            # TF-IDF vectors
            sp_path = os.path.join(path, 'cb_tfidf_vectors.npz')
            scipy.sparse.save_npz(sp_path, self.cb_filter.product_vectors_tfidf)

            # Count vectors (L2-normalized)
            cnt_path = os.path.join(path, 'cb_count_vectors.npz')
            if self.cb_filter.product_vectors_count is not None:
                scipy.sparse.save_npz(cnt_path, self.cb_filter.product_vectors_count)

            idx_path = os.path.join(path, 'product_id_to_idx.json')
            with open(idx_path, 'w') as f:
                json.dump(
                    {str(k): v for k, v in self.cb_filter.product_id_to_idx.items()},
                    f
                )
        
        print(f"Ensemble: Đã lưu tại {path}")
    
    @classmethod
    def load(cls, path: str = None, load_sub_models: bool = True):
        """
        Load ensemble model từ thư mục đã lưu.
        
        Args:
            path: đường dẫn thư mục (default: MODEL_DIR/ensemble)
            load_sub_models: nếu True, load luôn Item-CF, Item2Vec, Metapath2Vec
        
        Returns:
            EnsembleModel instance
        """
        if path is None:
            path = os.path.join(MODEL_DIR, "ensemble")
        
        # Load config
        with open(os.path.join(path, 'config.json')) as f:
            config = json.load(f)
        
        ensemble = cls(
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            top_k=config['top_k'],
            final_k=config['final_k'],
        )
        
        # Load CB Filter
        cb = CBFilter()
        tfidf_path = os.path.join(path, 'cb_tfidf_vectors.npz')
        if os.path.exists(tfidf_path):
            cb.product_vectors = scipy.sparse.load_npz(tfidf_path)
            # Load Count vectors nếu có
            cnt_path = os.path.join(path, 'cb_count_vectors.npz')
            if os.path.exists(cnt_path):
                cb.product_vectors_count = scipy.sparse.load_npz(cnt_path)
            idx_path = os.path.join(path, 'product_id_to_idx.json')
            with open(idx_path) as f:
                cb.product_id_to_idx = {int(k): v for k, v in json.load(f).items()}
            print(f"  CB Filter: {len(cb.product_id_to_idx)} products (TF-IDF + Count)")
        
        ensemble.cb_filter = cb
        
        if load_sub_models:
            # Load các sub-model từ MODEL_DIR
            from src.models.item_cf import ItemCFModel
            from src.models.item2vec import Item2VecModel
            from src.models.metapath2vec import Metapath2VecModel
            
            print("  Loading Item-CF...")
            item_cf = ItemCFModel()
            item_cf.load(os.path.join(MODEL_DIR, "item_cf"))
            
            print("  Loading Item2Vec...")
            i2v = Item2VecModel()
            i2v.load(os.path.join(MODEL_DIR, "item2vec"))
            
            print("  Loading Metapath2Vec...")
            mw = Metapath2VecModel()
            mw.load(os.path.join(MODEL_DIR, "metapath2vec"))
            
            ensemble.fit(item_cf, i2v, mw, cb)
        
        return ensemble