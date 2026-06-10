"""
Item2Vec: Word2Vec Skip-gram trên giỏ hàng.
Học embedding cho từng sản phẩm từ ngữ cảnh mua hàng.
"""
import os
import json
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd

from src.config import (
    I2V_VECTOR_SIZE, I2V_WINDOW, I2V_MIN_COUNT,
    I2V_NEGATIVE, I2V_EPOCHS, I2V_WORKERS, I2V_TOP_K
)


class Item2VecModel:
    """
    Item2Vec: Word2Vec Skip-gram trên giỏ hàng.
    
    Mỗi đơn hàng là 1 "câu", mỗi sản phẩm là 1 "từ".
    """
    
    def __init__(self, vector_size=None, window=None, min_count=None,
                 negative=None, epochs=None, workers=None):
        self.params = {
            'vector_size': vector_size if vector_size is not None else I2V_VECTOR_SIZE,
            'window': window if window is not None else I2V_WINDOW,
            'min_count': min_count if min_count is not None else I2V_MIN_COUNT,
            'negative': negative if negative is not None else I2V_NEGATIVE,
            'epochs': epochs if epochs is not None else I2V_EPOCHS,
            'workers': workers if workers is not None else I2V_WORKERS,
        }
        self.model = None  # gensim.models.Word2Vec
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
    
    def fit(self, order_products: pd.DataFrame, products_df: pd.DataFrame):
        """
        Train Word2Vec trên các order.
        
        Args:
            order_products: DataFrame [order_id, product_id, ...]
            products_df: DataFrame [product_id, ...]
        """
        print("Item2Vec: Bắt đầu fit...")
        
        # Mapping product_id → str (gensim yêu cầu string tags)
        all_product_ids = sorted(products_df['product_id'].unique())
        self.product_id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        self.idx_to_product_id = {i: pid for pid, i in self.product_id_to_idx.items()}
        
        # Tạo list of sentences (mỗi order là 1 list product_id dạng string)
        print("  Đang tạo sentences từ orders...")
        sentences = []
        grouped = order_products.groupby('order_id')['product_id']
        
        for order_id, group in grouped:
            items = [str(pid) for pid in group if pid in self.product_id_to_idx]
            if len(items) >= 2:  # Order phải có ít nhất 2 items
                sentences.append(items)
        
        print(f"  Số orders (sentences): {len(sentences)}")
        
        # Train Word2Vec
        print(f"  Đang train Word2Vec (size={self.params['vector_size']}, "
              f"window={self.params['window']}, epochs={self.params['epochs']})...")
        
        # Callback để log loss
        class LossLogger(CallbackAny2Vec):
            def __init__(self):
                self.epoch = 0
            
            def on_epoch_end(self, model):
                self.epoch += 1
                print(f"    Epoch {self.epoch}/{model.epochs}")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.params['vector_size'],
            window=self.params['window'],
            min_count=self.params['min_count'],
            negative=self.params['negative'],
            epochs=self.params['epochs'],
            workers=self.params['workers'],
            sg=1,  # Skip-gram
            seed=42,
            callbacks=[LossLogger()]
        )
        
        print(f"  Từ vựng: {len(self.model.wv)} products")
        print("Item2Vec: Fit hoàn tất.")
    
    def recommend(self, product_id: int, top_k: int = None):
        """
        Lấy top-K sản phẩm gần nhất với product_id trong không gian embedding — gợi ý mua kèm.

        Args:
            product_id: int — ID sản phẩm đầu vào
            top_k: int — số lượng gợi ý

        Returns:
            list (product_id, cosine_similarity) — các sản phẩm gợi ý kèm similarity
        """
        if top_k is None:
            top_k = I2V_TOP_K
        
        pid_str = str(product_id)
        if pid_str not in self.model.wv:
            return []
        
        try:
            similar = self.model.wv.most_similar(pid_str, topn=top_k)
            result = [(int(pid), float(sim)) for pid, sim in similar]
            return result
        except KeyError:
            return []
    
    def save(self, path: str):
        """
        Lưu model ra file (word2vec model + mapping).

        Args:
            path: đường dẫn thư mục đầu ra
        """
        os.makedirs(path, exist_ok=True)
        
        # Lưu gensim model
        self.model.save(os.path.join(path, "word2vec.model"))
        
        # Lưu mapping
        mapping = {
            'product_id_to_idx': {str(k): int(v) for k, v in self.product_id_to_idx.items()},
            'idx_to_product_id': {str(k): int(v) for k, v in self.idx_to_product_id.items()},
        }
        with open(os.path.join(path, "mapping.json"), 'w') as f:
            json.dump(mapping, f)
        
        print(f"Item2Vec: Đã lưu tại {path}")
    
    def load(self, path: str):
        """
        Load model từ file (word2vec model + mapping).

        Args:
            path: đường dẫn thư mục đã lưu
        """
        self.model = Word2Vec.load(os.path.join(path, "word2vec.model"))
        
        with open(os.path.join(path, "mapping.json"), 'r') as f:
            mapping = json.load(f)
        
        self.product_id_to_idx = {int(k): int(v) for k, v in mapping['product_id_to_idx'].items()}
        self.idx_to_product_id = {int(k): int(v) for k, v in mapping['idx_to_product_id'].items()}
        
        print(f"Item2Vec: Đã load từ {path}")
        print(f"  Từ vựng: {len(self.model.wv)}")