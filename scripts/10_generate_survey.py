"""
10 — Tạo file survey cho LLM đánh giá.
Với mỗi target product (10K), union top-10 từ 3 model gốc:
Item-CF, Item2Vec, KGMetapath.

Ensemble không được dùng vì nó chỉ weighted sum của 3 model trên,
không tự sinh candidate mới.

Output: data/survey/survey_samples.csv (6 cột)
  product_A_id, product_A_name, product_A_dept, product_B_id, product_B_name, product_B_dept

Cách chọn target:
  - Pool safe: product có count >= 10 và cả 3 model recommend được
  - 75% top popular (count cao nhất)
  - 25% random từ pool safe

Cách dùng:
  1. Chạy script này sau khi đã train xong 01→07
  2. Đưa file survey cho LLM, yêu cầu LLM đánh giá từng cặp
  3. LLM trả về file có thêm cột llm_label (1=complementary, 0=not)
  4. Dùng file đó làm ground truth để đánh giá từng model
"""
import os
import sys
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_SEED

N_TARGETS = 10_000               # tổng số target product
TOP_POPULAR_RATIO = 1        # 100% top bán chạy
RANDOM_RATIO = 0             # 0% random từ pool safe
TOP_K = 10                      # top-10 gợi ý từ mỗi model
MIN_COUNT = 10                  # ngưỡng safe: product phải xuất hiện >= 10 lần
SURVEY_DIR = os.path.join(MODEL_DIR, "..", "data", "survey")
OUTPUT_FILE = os.path.join(SURVEY_DIR, "survey_samples.csv")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_products():
    """Load sản phẩm (tên tiếng Việt)."""
    products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
    return products


def load_all_models():
    """Load tất cả model đã train. Nếu thiếu model nào → báo lỗi rõ ràng."""
    models = {}
    missing = []

    # 1. Item-CF
    try:
        from src.models.item_cf import ItemCFModel
        item_cf = ItemCFModel()
        item_cf.load(os.path.join(MODEL_DIR, "item_cf"))
        models['item_cf'] = item_cf
        print("  ✅ Item-CF loaded")
    except Exception as e:
        missing.append(f"Item-CF (chạy scripts/03_item_cf.py): {e}")

    # 2. Item2Vec (Neural CF)
    try:
        from src.models.item_cf_neural import ItemCFNeuralModel
        i2v = ItemCFNeuralModel()
        i2v.load(os.path.join(MODEL_DIR, "item2vec"))
        models['item2vec'] = i2v
        print("  ✅ Item2Vec loaded")
    except Exception as e:
        missing.append(f"Item2Vec (chạy scripts/04_item_cf_neural.py): {e}")

    # 3. KGMetapath
    try:
        from src.models.kg_metapath import KGMetapathModel
        mw = KGMetapathModel()
        mw.load(os.path.join(MODEL_DIR, "kg_metapath"))
        models['kg_metapath'] = mw
        print("  ✅ KGMetapath loaded")
    except Exception as e:
        missing.append(f"KGMetapath (chạy scripts/05_kg_metapath.py): {e}")

    if missing:
        print("\n❌ THIẾU MODEL — cần chạy các script sau:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    return models


def get_product_counts(models):
    """Lấy product_counts từ Item-CF metadata để biết top popular."""
    item_cf = models['item_cf']
    product_counts = item_cf.product_counts  # array (n_products,)
    idx_to_pid = item_cf.idx_to_product_id
    return product_counts, idx_to_pid


def build_safe_pool(models, product_counts, idx_to_pid):
    """
    Pool safe: product có count >= MIN_COUNT,
    và cả 3 model đều recommend được (có embedding / có trong từ điển).
    """
    safe_pids = set()
    item_cf = models['item_cf']
    i2v = models['item2vec']
    mw = models['kg_metapath']

    n_products = len(product_counts)
    for idx in range(n_products):
        pid = idx_to_pid[idx]
        if product_counts[idx] < MIN_COUNT:
            continue
        # Kiểm tra cả 3 model
        if pid not in item_cf.product_id_to_idx:
            continue
        if str(pid) not in i2v.model.wv:
            continue
        if pid not in mw.product_id_to_idx:
            continue
        safe_pids.add(pid)

    return sorted(safe_pids)


def get_candidates_for_product(models, product_id, top_k=TOP_K):
    """
    Union top-K từ tất cả model.
    Trả về set các product_id candidate.
    Dùng recommend() trực tiếp (không qua get_raw_candidates).
    """
    candidates = set()

    # Item-CF
    try:
        recs = models['item_cf'].recommend(product_id, top_k=top_k)
        candidates.update(pid for pid, _ in recs)
    except Exception:
        pass

    # Item2Vec
    try:
        recs = models['item2vec'].recommend(product_id, top_k=top_k)
        candidates.update(pid for pid, _ in recs)
    except Exception:
        pass

    # KGMetapath
    try:
        recs = models['kg_metapath'].recommend(product_id, top_k=top_k)
        candidates.update(pid for pid, _ in recs)
    except Exception:
        pass

    return candidates


def main():
    print("=" * 60)
    print("SCRIPT 10: TẠO FILE SURVEY CHO LLM ĐÁNH GIÁ")
    print("=" * 60)

    # 1. Load dữ liệu sản phẩm
    print("\n📦 Đang load sản phẩm...")
    products = load_products()
    pid_to_name = dict(zip(products['product_id'], products['product_name']))
    pid_to_dept = dict(zip(products['product_id'], products['department']))
    print(f"  Tổng sản phẩm: {len(products)}")

    # 2. Load tất cả model
    print("\n🧠 Đang load models...")
    models = load_all_models()

    # 3. Lấy product counts
    product_counts, idx_to_pid = get_product_counts(models)

    # 4. Xây pool safe (chỉ food products)
    print("\n🔒 Đang xây pool safe...")
    safe_pids = build_safe_pool(models, product_counts, idx_to_pid)
    print(f"  Pool safe: {len(safe_pids):,} sản phẩm")

    if len(safe_pids) < N_TARGETS:
        print(f"  ⚠️ Pool safe ({len(safe_pids):,}) < N_TARGETS ({N_TARGETS:,})")
        print(f"  → Dùng toàn bộ pool safe ({len(safe_pids):,})")
        n_targets = len(safe_pids)
    else:
        n_targets = N_TARGETS

    # 5. Chọn target
    print(f"\n🎯 Đang chọn {n_targets:,} target product...")
    
    # Sắp xếp pool safe theo count giảm dần
    count_of_pid = {pid: product_counts[idx_to_pid[pid]] 
                    for pid in safe_pids}
    safe_sorted = sorted(safe_pids, key=lambda pid: count_of_pid[pid], reverse=True)

    n_popular = int(n_targets * TOP_POPULAR_RATIO)
    n_random = n_targets - n_popular

    # 75% top popular
    popular_targets = safe_sorted[:n_popular]
    
    # 25% random từ phần còn lại
    remaining = [pid for pid in safe_sorted[n_popular:] if pid not in popular_targets]
    random_targets = random.sample(remaining, min(n_random, len(remaining)))
    
    targets = popular_targets + random_targets
    random.shuffle(targets)  # shuffle để không bị theo thứ tự popular/random
    
    print(f"  Top popular (75%): {len(popular_targets):,}")
    print(f"  Random (25%):     {len(random_targets):,}")
    print(f"  Tổng target:      {len(targets):,}")

    # 6. Lấy candidate cho từng target
    print(f"\n🔄 Đang lấy candidates cho {len(targets):,} targets...")
    print(f"  (Mỗi target union top-{TOP_K} từ 3 model, có thể mất thời gian)")
    
    rows = []
    for i, pid_a in enumerate(targets):
        if (i + 1) % 500 == 0:
            print(f"  Đã xử lý {i+1:,}/{len(targets):,} targets...")

        candidates = get_candidates_for_product(models, pid_a, top_k=TOP_K)
        
        if not candidates:
            continue

        name_a = pid_to_name.get(pid_a, f"Unknown({pid_a})")
        
        # Shuffle candidates để không lộ thứ tự model
        candidates_list = list(candidates)
        random.shuffle(candidates_list)

        for pid_b in candidates_list:
            name_b = pid_to_name.get(pid_b, f"Unknown({pid_b})")
            rows.append({
                'product_A_id': pid_a,
                'product_A_name': name_a,
                'product_A_dept': pid_to_dept.get(pid_a, ''),
                'product_B_id': pid_b,
                'product_B_name': name_b,
                'product_B_dept': pid_to_dept.get(pid_b, ''),
            })

    # 7. Xuất file
    os.makedirs(SURVEY_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    print(f"\n✅ Đã tạo file survey:")
    print(f"   File: {OUTPUT_FILE}")
    print(f"   Số dòng: {len(df):,}")
    print(f"   Số target có gợi ý: {df['product_A_id'].nunique():,}")
    print(f"   Trung bình gợi ý/target: {len(df) / df['product_A_id'].nunique():.1f}")
    print(f"\n📋 Cấu trúc file: product_A_id, product_A_name, product_A_dept, product_B_id, product_B_name, product_B_dept")
    print("\n👉 Cách dùng tiếp theo:")
    print("   1. Đưa file survey cho LLM")
    print("   2. Yêu cầu LLM đánh giá từng cặp (A,B): complementary (1) hay không (0)")
    print("   3. LLM trả về file có thêm cột llm_label")
    print("   4. Dùng file đó làm ground truth để tính metrics cho từng model")


if __name__ == "__main__":
    main()