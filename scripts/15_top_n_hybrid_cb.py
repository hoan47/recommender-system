"""
15 — Lấy top n sản phẩm được gợi ý nhiều nhất từ HYBRID (có CB filter).
Dựa trên ground truth từ gemini_responses_filtered.csv:
  - Với mỗi product_A, chạy ensemble.recommend() để lấy top-K gợi ý
  - Nếu product_B (ground truth) nằm trong top-K → tính là 1 lần gợi ý đúng
  - Đếm tần suất mỗi product_B xuất hiện → top n sản phẩm "ngon nhất"

Cách dùng:
   python scripts/15_top_n_hybrid_cb.py

Output:
   results/top_n_hybrid_cb.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR, RESULT_DIR, ENS_TOP_K
from src.models.ensemble import EnsembleModel


def main():
    n = 100  # số lượng top sản phẩm muốn lấy, có thể đổi

    print("=" * 60)
    print("  SCRIPT 15: TOP N SẢN PHẨM NGON NHẤT TỪ HYBRID + CB")
    print("=" * 60)

    # 1. Load ensemble model
    print("\n📦 Loading ensemble model...")
    ensemble = EnsembleModel.load(load_sub_models=True)
    print(f"  ✅ Loaded ensemble (α={ensemble.alpha}, β={ensemble.beta}, γ={ensemble.gamma})")

    # 2. Load ground truth
    print("\n📖 Loading ground truth...")
    gt_path = os.path.join(MODEL_DIR, "..", "data", "survey",
                           "llm_raw_responses", "gemini_responses_filtered.csv")
    df_gt = pd.read_csv(gt_path, encoding='utf-8')
    print(f"  Total pairs: {len(df_gt):,}")

    # 3. Load product names
    print("\n🏷️  Loading product names...")
    products_path = os.path.join(MODEL_DIR, "..", "data", "processed", "products_vi.csv")
    df_products = pd.read_csv(products_path, encoding='utf-8')
    product_name_map = dict(zip(df_products['product_id'], df_products['product_name_vi']))
    print(f"  Total products: {len(product_name_map):,}")

    # 4. Lấy unique product_A_id
    unique_a = sorted(df_gt['product_A_id'].unique())
    print(f"\n🎯 Unique product_A: {len(unique_a):,}")

    # 5. Với mỗi product_A, recommend và kiểm tra ground truth
    print(f"\n🔮 Running HYBRID recommendation (top_k={ENS_TOP_K}, use_cb_filter=True)...")
    
    # Dict đếm: {product_B_id: số lần được recommend đúng}
    hit_count = {}
    total_pairs_checked = 0
    n_cold_start = 0  # product_A không có trong model

    for i, pid_a in enumerate(unique_a):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(unique_a)}")

        # Lấy ground truth product_B cho pid_a này
        gt_b_ids = set(df_gt[df_gt['product_A_id'] == pid_a]['product_B_id'].tolist())

        # Gọi ensemble recommend
        try:
            recs = ensemble.recommend(pid_a, use_cb_filter=True, top_k=ENS_TOP_K)
        except Exception as e:
            print(f"  ⚠️  Lỗi khi recommend product {pid_a}: {e}")
            n_cold_start += 1
            continue

        if not recs:
            n_cold_start += 1
            continue

        # Lấy set product_id từ recommendations
        rec_product_ids = {pid for pid, score in recs}

        # Kiểm tra ground truth product_B có trong recommendations không
        for pid_b in gt_b_ids:
            total_pairs_checked += 1
            if pid_b in rec_product_ids:
                hit_count[pid_b] = hit_count.get(pid_b, 0) + 1

    # 6. Sắp xếp và lấy top n
    print(f"\n📊 Tổng số cặp đã kiểm tra: {total_pairs_checked:,}")
    print(f"   Product_A cold-start (không recommend được): {n_cold_start}")
    print(f"   Số product_B trúng ít nhất 1 lần: {len(hit_count):,}")

    sorted_hits = sorted(hit_count.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_hits[:n]

    # 7. Tạo DataFrame kết quả
    results = []
    for rank, (pid, freq) in enumerate(top_n, 1):
        name = product_name_map.get(pid, "Không rõ")
        results.append({
            'rank': rank,
            'product_id': pid,
            'product_name_vi': name,
            'frequency': freq,
        })

    df_result = pd.DataFrame(results)

    # 8. In ra console
    print("\n" + "=" * 60)
    print(f"  TOP {n} SẢN PHẨM NGON NHẤT (HYBRID + CB)")
    print("=" * 60)
    for _, row in df_result.iterrows():
        print(f"  #{row['rank']:>2} | ID {row['product_id']:>6} | freq={row['frequency']:>3} | {row['product_name_vi']}")

    # 9. Lưu kết quả
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, "top_n_hybrid_cb.csv")
    df_result.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Đã lưu kết quả tại: {output_path}")

    return output_path


if __name__ == "__main__":
    main()