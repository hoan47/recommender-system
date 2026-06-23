"""
12 — Lọc ground truth Gemini bằng CB similarity.
Xóa các cặp substitute (CB similarity >= threshold) khỏi gemini_responses.csv
để đánh giá model chính xác hơn (không bị trừ điểm oan).

Cách dùng:
   python scripts/12_filter_gemini_cb.py

Yêu cầu:
   - Đã chạy scripts/02_cb_filter.py (có models/cb_filter/)
   - File data/survey/llm_raw_responses/gemini_responses.csv tồn tại

Output:
   data/survey/llm_raw_responses/gemini_responses_filtered.csv
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR
from src.features.vectorizer import cb_ensemble_similarity


def load_cb_vectors():
    """Load CB vectors và mapping product_id → idx."""
    cb_dir = os.path.join(MODEL_DIR, "cb_filter")

    tfidf_vectors = scipy.sparse.load_npz(
        os.path.join(cb_dir, "tfidf_vectors.npz")
    )
    count_vectors = scipy.sparse.load_npz(
        os.path.join(cb_dir, "count_vectors.npz")
    )

    with open(os.path.join(cb_dir, "product_id_to_idx.json"), 'r') as f:
        product_id_to_idx = {int(k): v for k, v in json.load(f).items()}

    # Đọc alpha từ config đã dùng khi train
    # Mặc định CB_ALPHA = 0.5
    from src.config import CB_ALPHA
    alpha = CB_ALPHA

    print(f"  CB vectors loaded:")
    print(f"    TF-IDF: {tfidf_vectors.shape}")
    print(f"    Count:  {count_vectors.shape}")
    print(f"    Products: {len(product_id_to_idx)}")
    print(f"    Alpha: {alpha}")

    return tfidf_vectors, count_vectors, product_id_to_idx, alpha


def main():
    print("=" * 60)
    print("  SCRIPT 12: LỌC GROUND TRUTH GEMINI BẰNG CB SIMILARITY")
    print("=" * 60)

    # 1. Load CB vectors
    print("\n📦 Loading CB vectors...")
    tfidf_vectors, count_vectors, product_id_to_idx, alpha = load_cb_vectors()

    # 2. Load ground truth
    print("\n📖 Loading ground truth...")
    gt_path = os.path.join(MODEL_DIR, "..", "data", "survey",
                           "llm_raw_responses", "gemini_responses.csv")
    df = pd.read_csv(gt_path, encoding='utf-8')
    total_pairs = len(df)
    print(f"  Total pairs: {total_pairs:,}")

    # 3. Tính CB similarity cho từng cặp
    print("\n🔬 Computing CB similarity for each pair...")
    THRESHOLD = 0.3  # ngưỡng từ config ENS_CB_THRESHOLD

    keep_mask = []
    n_skipped = 0  # sản phẩm không có vector (cold-start)
    n_substitute = 0
    n_complementary = 0

    # Process theo batch để tận dụng vectorization
    # Chia theo product_A_id để tính similarity theo batch
    grouped = df.groupby('product_A_id')

    results = []
    for pid_a, group in grouped:
        pids_b = group['product_B_id'].tolist()
        descs = group['description'].tolist()

        if pid_a not in product_id_to_idx:
            # Cold-start: giữ nguyên tất cả cặp
            for pid_b, desc in zip(pids_b, descs):
                results.append({'product_A_id': pid_a,
                                'product_B_id': pid_b,
                                'description': desc})
                n_skipped += 1
            continue

        idx_a = product_id_to_idx[pid_a]

        # Tìm các product_B có vector
        valid_indices = []
        valid_b_ids = []
        valid_descs = []
        cold_b_ids = []
        cold_descs = []

        for pid_b, desc in zip(pids_b, descs):
            if pid_b in product_id_to_idx:
                valid_indices.append(product_id_to_idx[pid_b])
                valid_b_ids.append(pid_b)
                valid_descs.append(desc)
            else:
                cold_b_ids.append(pid_b)
                cold_descs.append(desc)

        # Tính similarity cho các cặp có vector
        if valid_indices:
            sims = cb_ensemble_similarity(
                tfidf_vectors, count_vectors,
                idx_a, valid_indices, alpha=alpha,
            )

            for pid_b, desc, sim in zip(valid_b_ids, valid_descs, sims):
                if sim >= THRESHOLD:
                    n_substitute += 1
                    # Bỏ qua — substitute
                else:
                    results.append({'product_A_id': pid_a,
                                    'product_B_id': pid_b,
                                    'description': desc})
                    n_complementary += 1

        # Cold-start: giữ nguyên
        for pid_b, desc in zip(cold_b_ids, cold_descs):
            results.append({'product_A_id': pid_a,
                            'product_B_id': pid_b,
                            'description': desc})
            n_skipped += 1

    # 4. Lưu kết quả
    output_path = os.path.join(MODEL_DIR, "..", "data", "survey",
                               "llm_raw_responses", "gemini_responses_filtered.csv")
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False, encoding='utf-8')

    # 5. In thống kê
    print("\n" + "=" * 60)
    print("  KẾT QUẢ LỌC")
    print("=" * 60)
    print(f"  Tổng số cặp ban đầu:      {total_pairs:>6,}")
    print(f"  Substitute (bị loại):     {n_substitute:>6,}")
    print(f"  Cold-start (giữ lại):     {n_skipped:>6,}")
    print(f"  Complementary (giữ lại):  {n_complementary:>6,}")
    print(f"  Tổng sau lọc:             {len(df_out):>6,}")
    print(f"  Ngưỡng CB threshold:      {THRESHOLD}")
    print(f"\n✅ Đã lưu tại: {output_path}")

    return output_path


if __name__ == "__main__":
    main()