"""
Pipeline chạy toàn bộ các model recommendation.
Chạy: python run.py

Các bước:
  1. Load dữ liệu
  2. CB Filter — vector hóa sản phẩm
  3. Ochiai + Confidence Score — xây co-occurrence matrix
  4. Item2Vec — Word2Vec trên giỏ hàng
  5. Node2Vec — graph embedding
  6. Association Rules — từ co-occurrence matrix
  7. Ensemble — weighted score + optional CB Filter
  8. Test thử với vài sản phẩm mẫu
"""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_DIR
from src.features.loader import load_products, load_order_products
from src.models.cb_filter import CBFilter
from src.models.ochiai import OchiaiModel
from src.models.item2vec import Item2VecModel
from src.models.node2vec import Node2VecModel
from src.models.assoc_rules import AssocRulesModel
from src.models.ensemble import EnsembleModel


def log_step(step, msg):
    print(f"\n{'='*60}")
    print(f"  BƯỚC {step}: {msg}")
    print(f"{'='*60}")


def main():
    start_time = time.time()
    
    # ============================================================
    # Bước 1: Load dữ liệu
    # ============================================================
    log_step(1, "Load dữ liệu")
    
    print("  Đang load products...")
    products = load_products()
    print(f"  Products: {len(products)} sản phẩm")
    
    print("  Đang load order_products (prior + train)...")
    order_products = load_order_products(use_prior=True, use_train=True)
    print(f"  Orders: {len(order_products)} records")
    print(f"  Unique orders: {order_products['order_id'].nunique()}")
    
    # ============================================================
    # Bước 2: CB Filter
    # ============================================================
    log_step(2, "CB Filter — Content-Based Diversity Filter")
    
    cb_path = os.path.join(MODEL_DIR, "cb_filter")
    if os.path.exists(os.path.join(cb_path, "product_vectors.npz")):
        print("  CB Filter đã train, skip...")
        # CBFilter sẽ fit lại từ products
    else:
        print("  Đang fit CB Filter...")
    
    cb = CBFilter()
    cb.fit(products)
    # Lưu product vectors
    os.makedirs(cb_path, exist_ok=True)
    import scipy.sparse
    scipy.sparse.save_npz(os.path.join(cb_path, "product_vectors.npz"),
                          cb.product_vectors)
    print(f"  CB Filter hoàn tất. {len(cb.product_id_to_idx)} sản phẩm đã vector hóa.")
    
    # ============================================================
    # Bước 3: Ochiai + Confidence
    # ============================================================
    log_step(3, "Ochiai + Confidence Score")
    
    ochiai_path = os.path.join(MODEL_DIR, "ochiai")
    if os.path.exists(os.path.join(ochiai_path, "cooc_matrix.npz")):
        print("  OchiaiModel đã train, loading...")
        ochiai = OchiaiModel()
        ochiai.load(ochiai_path)
    else:
        print("  Đang fit OchiaiModel (có thể mất vài phút)...")
        ochiai = OchiaiModel()
        ochiai.fit(order_products, products)
        ochiai.save(ochiai_path)
    
    # Test thử
    sample_id = products['product_id'].iloc[0]
    print(f"\n  Test recommend cho product_id={sample_id}:")
    recs = ochiai.recommend(sample_id, top_k=5)
    for pid, score in recs:
        pname = products[products['product_id'] == pid]['product_name'].values
        pname = pname[0] if len(pname) else "?"
        print(f"    → {pid}: {pname} (score={score:.4f})")
    
    # ============================================================
    # Bước 4: Item2Vec
    # ============================================================
    log_step(4, "Item2Vec — Word2Vec trên giỏ hàng")
    
    i2v_path = os.path.join(MODEL_DIR, "item2vec")
    if os.path.exists(os.path.join(i2v_path, "word2vec.model")):
        print("  Item2VecModel đã train, loading...")
        i2v = Item2VecModel()
        i2v.load(i2v_path)
    else:
        print("  Đang fit Item2VecModel (có thể mất vài phút)...")
        i2v = Item2VecModel()
        i2v.fit(order_products, products)
        i2v.save(i2v_path)
    
    # Test thử
    print(f"\n  Test recommend cho product_id={sample_id}:")
    recs = i2v.recommend(sample_id, top_k=5)
    for pid, sim in recs:
        pname = products[products['product_id'] == pid]['product_name'].values
        pname = pname[0] if len(pname) else "?"
        print(f"    → {pid}: {pname} (sim={sim:.4f})")
    
    # ============================================================
    # Bước 5: Node2Vec
    # ============================================================
    log_step(5, "Node2Vec — Graph embedding")
    
    n2v_path = os.path.join(MODEL_DIR, "node2vec")
    if os.path.exists(os.path.join(n2v_path, "embeddings.npy")):
        print("  Node2VecModel đã train, loading...")
        n2v = Node2VecModel()
        n2v.load(n2v_path)
    else:
        print("  Đang fit Node2VecModel (có thể mất nhiều phút)...")
        n2v = Node2VecModel()
        n2v.fit(order_products, products)
        n2v.save(n2v_path)
    
    # Test thử
    print(f"\n  Test recommend cho product_id={sample_id}:")
    recs = n2v.recommend(sample_id, top_k=5)
    for pid, sim in recs:
        pname = products[products['product_id'] == pid]['product_name'].values
        pname = pname[0] if len(pname) else "?"
        print(f"    → {pid}: {pname} (sim={sim:.4f})")
    
    # ============================================================
    # Bước 6: Association Rules
    # ============================================================
    log_step(6, "Association Rules — từ co-occurrence matrix")
    
    arm_path = os.path.join(MODEL_DIR, "assoc_rules")
    if os.path.exists(os.path.join(arm_path, "rules.csv")):
        print("  AssocRulesModel đã train, loading...")
        arm = AssocRulesModel()
        arm.load(arm_path)
    else:
        print("  Đang fit AssocRulesModel...")
        arm = AssocRulesModel()
        arm.fit(ochiai, order_products)
        arm.save(arm_path)
    
    # Test thử
    print(f"\n  Test recommend cho product_id={sample_id}:")
    recs = arm.recommend(sample_id, top_k=5)
    for pid, lift in recs:
        pname = products[products['product_id'] == pid]['product_name'].values
        pname = pname[0] if len(pname) else "?"
        print(f"    → {pid}: {pname} (lift={lift:.4f})")
    
    # ============================================================
    # Bước 7: Ensemble
    # ============================================================
    log_step(7, "Ensemble — Co-occurrence Ensemble + CB Filter")
    
    ensemble = EnsembleModel()
    ensemble.fit(ochiai, i2v, n2v, cb)
    
    # ============================================================
    # Bước 8: Test thử với vài sản phẩm mẫu
    # ============================================================
    log_step(8, "Test thử recommendations")
    
    test_products = [
        1,          # Chocolate
        250,        # (random sản phẩm)
        5000,       # (random sản phẩm)
        25000,      # (random sản phẩm)
    ]
    
    for pid in test_products:
        pname = products[products['product_id'] == pid]['product_name'].values
        pname = pname[0] if len(pname) else f"Product {pid}"
        
        print(f"\n  Sản phẩm đầu vào: [{pid}] {pname}")
        
        # Ensemble không CB Filter
        recs_no_cb = ensemble.recommend(pid, use_cb_filter=False)
        print(f"  Without CB Filter (top-5):")
        for rid, score in recs_no_cb[:5]:
            rname = products[products['product_id'] == rid]['product_name'].values
            rname = rname[0] if len(rname) else "?"
            print(f"    → {rid}: {rname} (score={score:.4f})")
        
        # Ensemble có CB Filter
        recs_cb = ensemble.recommend(pid, use_cb_filter=True)
        print(f"  With CB Filter (top-5):")
        for rid, score in recs_cb[:5]:
            rname = products[products['product_id'] == rid]['product_name'].values
            rname = rname[0] if len(rname) else "?"
            print(f"    → {rid}: {rname} (score={score:.4f})")
    
    # ============================================================
    # Kết thúc
    # ============================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  PIPELINE HOÀN TẤT!")
    print(f"  Thời gian: {elapsed:.1f} giây ({elapsed/60:.1f} phút)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()