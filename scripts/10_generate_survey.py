"""
10 — Tạo file survey cho LLM đánh giá.
Với mỗi target product (10K), union top-10 từ 5 model:
Item-CF, Item2Vec, KGMetapath, Ensemble, Ensemble+CB.

Output: data/survey/survey_samples.csv (4 cột)
  product_A_id, product_A_name, product_B_id, product_B_name

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

from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_SEED, DEPARTMENTS_FILE, PRODUCTS_FILE


# ============================================================
# Cấu hình
# ============================================================
EXCLUDED_DEPARTMENT_NAMES = ['other', 'pets', 'personal care', 'household', 'babies', 'missing']
# other — không phân loại rõ ràng
# pets — thức ăn, phụ kiện thú cưng
# personal care — sữa tắm, dầu gội, kem đánh răng, mỹ phẩm
# household — đồ gia dụng, bột giặt, giấy vệ sinh, túi nilon
# babies — tã bỉm, khăn ướt, đồ chơi, phụ kiện em bé
# missing — dữ liệu lỗi/thiếu thông tin ngành hàng

N_TARGETS = 5_000               # tổng số target product
TOP_POPULAR_RATIO = 0.75        # 75% top bán chạy
RANDOM_RATIO = 0.25             # 25% random từ pool safe
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


def get_food_product_ids():
    """
    Xác định product_id thuộc Food & Beverage dựa trên department name.
    
    Đọc departments.csv để map tên department → department_id,
    sau đó đọc products.csv gốc để lấy product_id thuộc các department
    KHÔNG nằm trong EXCLUDED_DEPARTMENT_NAMES.
    
    Returns:
        set[int] — các product_id thuộc thực phẩm (food)
    """
    departments = pd.read_csv(DEPARTMENTS_FILE)
    excluded_dept_ids = set(
        departments[departments['department'].isin(EXCLUDED_DEPARTMENT_NAMES)]['department_id']
    )
    
    products_raw = pd.read_csv(PRODUCTS_FILE)
    food_mask = ~products_raw['department_id'].isin(excluded_dept_ids)
    food_pids = set(products_raw.loc[food_mask, 'product_id'])
    
    print(f"  Tổng sản phẩm (products.csv): {len(products_raw):,}")
    print(f"  Food products: {len(food_pids):,} ({(len(food_pids)/len(products_raw))*100:.1f}%)")
    print(f"  Non-Food bị loại: {len(products_raw) - len(food_pids):,} "
          f"({(1 - len(food_pids)/len(products_raw))*100:.1f}%)")
    
    return food_pids


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

    # 4. Ensemble + CB
    try:
        from src.models.ensemble import EnsembleModel
        ensemble = EnsembleModel.load(load_sub_models=False)
        models['ensemble'] = ensemble
        print("  ✅ Ensemble loaded")
    except Exception as e:
        missing.append(f"Ensemble (chạy scripts/07_ensemble.py): {e}")

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


def build_safe_pool(models, product_counts, idx_to_pid, food_pids):
    """
    Pool safe: product có count >= MIN_COUNT, thuộc food,
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
        # Chỉ giữ food products
        if pid not in food_pids:
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


def recommend_ensemble_with_topk(ensemble, product_id, top_k, use_cb_filter):
    """Wrapper: set final_k + top_k tạm thời để lấy đúng số lượng gợi ý."""
    orig_final = ensemble.final_k
    orig_top = ensemble.top_k
    ensemble.final_k = top_k
    if top_k > ensemble.top_k:
        ensemble.top_k = top_k
    try:
        return ensemble.recommend(product_id, use_cb_filter=use_cb_filter)
    finally:
        ensemble.final_k = orig_final
        ensemble.top_k = orig_top


def get_candidates_for_product(models, product_id, top_k=TOP_K, food_pids=None):
    """
    Union top-K từ tất cả model.
    Chỉ giữ candidate thuộc food_pids (nếu được cung cấp).
    Trả về set các product_id candidate.
    Phiên bản này dùng recommend() trực tiếp (không qua get_raw_candidates).
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

    # Ensemble (w/o CB) — dùng wrapper
    try:
        recs = recommend_ensemble_with_topk(
            models['ensemble'], product_id, top_k, use_cb_filter=False
        )
        candidates.update(pid for pid, _ in recs)
    except Exception:
        pass

    # Ensemble + CB
    try:
        recs = recommend_ensemble_with_topk(
            models['ensemble'], product_id, top_k, use_cb_filter=True
        )
        candidates.update(pid for pid, _ in recs)
    except Exception:
        pass

    # Lọc chỉ giữ food candidates
    if food_pids is not None:
        candidates &= food_pids

    return candidates


def main():
    print("=" * 60)
    print("SCRIPT 10: TẠO FILE SURVEY CHO LLM ĐÁNH GIÁ")
    print("=" * 60)

    # 1. Load dữ liệu sản phẩm
    print("\n📦 Đang load sản phẩm...")
    products = load_products()
    pid_to_name = dict(zip(products['product_id'], products['product_name']))
    print(f"  Tổng sản phẩm: {len(products)}")

    # 1b. Xác định food products
    print("\n🍏 Đang xác định Food & Beverage products...")
    food_pids = get_food_product_ids()

    # 2. Load tất cả model
    print("\n🧠 Đang load models...")
    models = load_all_models()

    # 3. Lấy product counts
    product_counts, idx_to_pid = get_product_counts(models)

    # 4. Xây pool safe (chỉ food products)
    print("\n🔒 Đang xây pool safe...")
    safe_pids = build_safe_pool(models, product_counts, idx_to_pid, food_pids)
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
    print(f"  (Mỗi target union top-{TOP_K} từ 5 model, có thể mất thời gian)")
    
    rows = []
    for i, pid_a in enumerate(targets):
        if (i + 1) % 500 == 0:
            print(f"  Đã xử lý {i+1:,}/{len(targets):,} targets...")

        candidates = get_candidates_for_product(models, pid_a, top_k=TOP_K, food_pids=food_pids)
        
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
                'product_B_id': pid_b,
                'product_B_name': name_b,
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
    print(f"\n📋 Cấu trúc file: product_A_id, product_A_name, product_B_id, product_B_name")
    print("\n👉 Cách dùng tiếp theo:")
    print("   1. Đưa file survey cho LLM")
    print("   2. Yêu cầu LLM đánh giá từng cặp (A,B): complementary (1) hay không (0)")
    print("   3. LLM trả về file có thêm cột llm_label")
    print("   4. Dùng file đó làm ground truth để tính metrics cho từng model")


if __name__ == "__main__":
    main()