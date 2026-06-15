"""
Association Rules (Baseline).
Tự implement từ co-occurrence matrix (ItemCFModel) — không cần FP-Growth.
Chạy được trên full dataset.
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT, ARM_TOP_K


class AssocRulesModel:
    """
    Association Rules — tự implement từ co-occurrence matrix.
    
    Công thức (dựa trên cooc matrix từ ItemCFModel):
      support(A,B)      = cooc[A,B] / total_orders
      confidence(A→B)   = cooc[A,B] / count(A)
      lift(A,B)         = confidence(A,B) / (count(B) / total_orders)
    """
    
    def __init__(self, min_support=None, min_confidence=None,
                 min_lift=None, top_k=None):
        self.min_support = min_support if min_support is not None else ARM_MIN_SUPPORT
        self.min_confidence = min_confidence if min_confidence is not None else ARM_MIN_CONFIDENCE
        self.min_lift = min_lift if min_lift is not None else ARM_MIN_LIFT
        self.top_k = top_k if top_k is not None else ARM_TOP_K
        
        self.total_orders = 0
        self.cooc_matrix = None        # CSR matrix (dùng chung với ItemCFModel)
        self.product_counts = None     # array (n_products,)
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self.rules_df = None           # DataFrame [antecedent, consequent, support, confidence, lift]
    
    def fit(self, item_cf_model, order_products: pd.DataFrame):
        """
        Xây rules từ co-occurrence matrix của ItemCFModel.

        Args:
            item_cf_model: ItemCFModel đã train (có cooc_matrix + product_counts)
            order_products: DataFrame [order_id, product_id, ...] (để tính total_orders)
        """
        print("AssocRules: Bắt đầu fit...")
        
        # Copy từ ItemCFModel
        self.cooc_matrix = item_cf_model.cooc_matrix
        self.product_counts = item_cf_model.product_counts
        self.product_id_to_idx = item_cf_model.product_id_to_idx
        self.idx_to_product_id = item_cf_model.idx_to_product_id
        self.total_orders = item_cf_model.total_orders
        
        n_products = self.cooc_matrix.shape[0]
        print(f"  Số sản phẩm: {n_products}")
        print(f"  Non-zero entries: {self.cooc_matrix.nnz}")
        print(f"  Total orders: {self.total_orders}")
        
        # Duyệt non-zero rows của CSR matrix
        print("  Đang tính rules...")
        rules_data = []
        cooc = self.cooc_matrix
        
        # Duyệt từng product A (có count > 0)
        for idx_a in tqdm(range(n_products), desc="  Products"):
            count_a = self.product_counts[idx_a]
            if count_a == 0:
                continue
            
            # Lấy row của A (các product B có co-occurrence)
            row = cooc[idx_a].toarray().flatten()
            nonzero_indices = np.where(row > 0)[0]
            
            for idx_b in nonzero_indices:
                cnt = row[idx_b]
                count_b = self.product_counts[idx_b]
                if count_b == 0:
                    continue
                
                # Tính metrics
                support = cnt / self.total_orders
                confidence = cnt / count_a
                # lift = confidence / (count_b / total_orders)
                support_b = count_b / self.total_orders
                lift = confidence / support_b if support_b > 0 else 0
                
                # Filter
                if (support >= self.min_support
                        and confidence >= self.min_confidence
                        and lift >= self.min_lift):
                    rules_data.append({
                        'antecedent': self.idx_to_product_id[idx_a],
                        'consequent': self.idx_to_product_id[idx_b],
                        'support': support,
                        'confidence': confidence,
                        'lift': lift,
                        'count': int(cnt),
                    })
        
        self.rules_df = pd.DataFrame(rules_data)
        print(f"  Tổng số rules (sau filter): {len(self.rules_df)}")
        
        # Sort theo lift descending
        if not self.rules_df.empty:
            self.rules_df = self.rules_df.sort_values('lift', ascending=False).reset_index(drop=True)
        
        print("AssocRules: Fit hoàn tất.")
    
    def recommend(self, product_id: int, top_k: int = None):
        """
        Tìm rules có antecedent = product_id — gợi ý mua kèm.
        Sort theo lift descending.

        Args:
            product_id: int — ID sản phẩm đầu vào
            top_k: int — số lượng gợi ý

        Returns:
            list (product_id, lift) — các sản phẩm gợi ý kèm lift score
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.rules_df is None or self.rules_df.empty:
            return []
        
        # Lọc rules có antecedent = product_id
        product_rules = self.rules_df[
            self.rules_df['antecedent'] == product_id
        ].copy()
        
        if product_rules.empty:
            return []
        
        # Sort và lấy top-K
        product_rules = product_rules.sort_values('lift', ascending=False)
        product_rules = product_rules.head(top_k)
        
        result = [
            (int(row['consequent']), float(row['lift']))
            for _, row in product_rules.iterrows()
        ]
        
        return result
    
    def save(self, path: str):
        """
        Lưu rules ra CSV + metadata.

        Args:
            path: đường dẫn thư mục đầu ra
        """
        os.makedirs(path, exist_ok=True)
        
        if self.rules_df is not None:
            filepath = os.path.join(path, "rules.csv")
            self.rules_df.to_csv(filepath, index=False)
            
            # Lưu metadata
            metadata = {
                'min_support': self.min_support,
                'min_confidence': self.min_confidence,
                'min_lift': self.min_lift,
                'total_orders': int(self.total_orders),
                'n_rules': len(self.rules_df),
                'product_id_to_idx': {str(k): int(v) for k, v in self.product_id_to_idx.items()},
                'idx_to_product_id': {str(k): int(v) for k, v in self.idx_to_product_id.items()},
            }
            with open(os.path.join(path, "metadata.json"), 'w') as f:
                json.dump(metadata, f)
            
            print(f"AssocRules: Đã lưu {len(self.rules_df)} rules tại {path}")
    
    def load(self, path: str):
        """
        Load rules từ CSV + metadata.

        Args:
            path: đường dẫn thư mục đã lưu
        """
        filepath = os.path.join(path, "rules.csv")
        self.rules_df = pd.read_csv(filepath)
        
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        self.min_support = metadata['min_support']
        self.min_confidence = metadata['min_confidence']
        self.min_lift = metadata['min_lift']
        self.total_orders = int(metadata['total_orders'])
        self.product_id_to_idx = {int(k): int(v) for k, v in metadata['product_id_to_idx'].items()}
        self.idx_to_product_id = {int(k): int(v) for k, v in metadata['idx_to_product_id'].items()}
        
        print(f"AssocRules: Đã load {len(self.rules_df)} rules từ {path}")