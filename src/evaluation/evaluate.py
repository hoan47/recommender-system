"""
Đánh giá chính xác cho bài toán basket completion / complementary recommendation

THIẾT KẾ DÙNG:
  - Ground truth = TẤT CẢ SP còn lại trong cùng order (không phân biệt dept)
  - Lý do: model gợi ý "mua kèm" -> người dùng muốn mua thêm, bất kỳ dept nào
  - Cross-dept ưu tiên là việc của MODEL (qua cross_bonus), không phải của EVALUATOR

TEST SPLIT:
  - User-based temporal: 80% order đầu làm train, 20% order cuối làm test
  - Đảm bảo không có data leakage
  - Mỗi test order có ~8-12 SP -> nhiều cơ hội hit

METRICS DÙNG:
  - H@K (Hit Rate): tỷ lệ order có >= 1 gợi ý đúng -> metric chính
  - P@K, R@K, F1@K, NDCG@K, MAP@K: bổ sung
  - Không dùng MAE/MSE/RMSE (không phù hợp với ranking task)

BENCHMARK THỰC TẾ:
  Với dataset 50k SP, sparse 99.9%:
    H@10 < 0.10  = còn thấp
    H@10 ~ 0.15  = chấp nhận được
    H@10 ~ 0.25  = tốt
    H@10 > 0.35  = rất tốt (ngang Amazon/Instacart production)
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import src.data_loader as dl
from src.config import PATH_OUTPUT_CSV


def calc_metrics(recommended: list,
                 ground_truth: list,
                 ks: tuple = (10, 50, 100)) -> dict:
    """Tính toán các metrics cho 1 query."""
    if not ground_truth or not recommended:
        return {}

    truth_set   = set(ground_truth)
    total_truth = len(truth_set)
    results     = {}

    for k in ks:
        k_int     = int(k)
        top_k     = recommended[:k_int]
        hits      = len(set(top_k) & truth_set)
        precision = hits / k_int
        recall    = hits / total_truth
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        hit_rate  = 1.0 if hits > 0 else 0.0
        ndcg      = _ndcg(top_k, truth_set, k_int)
        map_k     = _ap_at_k(top_k, truth_set, k_int)

        results[f'H@{k_int}']    = hit_rate
        results[f'NDCG@{k_int}'] = ndcg
        results[f'MAP@{k_int}']  = map_k
        results[f'P@{k_int}']    = precision
        results[f'R@{k_int}']    = recall
        results[f'F1@{k_int}']   = f1

    return results


def _ndcg(top_k: list, truth_set: set, k: int) -> float:
    """NDCG@K: tính đến vị trí xếp hạng của hit."""
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top_k) if p in truth_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def _ap_at_k(ranked_list: list, gt_set: set, k: int) -> float:
    """Average Precision @K."""
    if not gt_set:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    n_rel = min(len(gt_set), k)
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i] in gt_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / n_rel if n_rel > 0 else 0.0


def evaluate_one(rec_func, name: str,
                 ks: tuple = (10, 50, 100)) -> dict:
    """
    Đánh giá 1 model trên tất cả test cases.

    Tham số:
        rec_func: function(seed_product, top_k) -> list[int] recommendations
        name: str — tên model
        ks: tuple — các giá trị K

    Trả về:
        dict: {metric@K: mean_value, 'Time(s)', 'N_valid'}
    """
    metric_lists = defaultdict(list)
    t0 = time.time()

    for seed, truth in tqdm(dl.test_cases, desc=f"  [{name}]", ncols=80):
        if not truth:
            continue
        try:
            recs = rec_func(seed, max(ks))
            if not recs:
                continue
            m = calc_metrics(recs, truth, ks)
            for key, val in m.items():
                metric_lists[key].append(val)
        except Exception:
            continue

    elapsed = round(time.time() - t0, 2)
    if not metric_lists:
        r = {'Time(s)': elapsed}
        for k in ks:
            for p in ['H', 'NDCG', 'MAP', 'P', 'R', 'F1']:
                r[f'{p}@{k}'] = 0.0
        r['N_valid'] = 0
        return r

    result = {key: round(float(np.mean(vals)), 4)
              for key, vals in metric_lists.items()}
    result['Time(s)']  = elapsed
    result['N_valid']  = len(next(iter(metric_lists.values())))
    return result


def run_comparison(models_dict: dict,
                   ks: tuple = (10, 50, 100)) -> pd.DataFrame:

    # Ưu tiên H@K và NDCG@K trong col_order (metrics quan trọng nhất)
    col_order = []
    for k in ks:
        k_int = int(k)
        col_order.extend([
            f'H@{k_int}', f'NDCG@{k_int}', f'MAP@{k_int}',
            f'P@{k_int}', f'R@{k_int}', f'F1@{k_int}'
        ])
    col_order += ['Time(s)', 'N_valid']

    # Thống kê test set
    n_cases  = len(dl.test_cases)
    avg_truth = np.mean([len(t) for _, t in dl.test_cases]) if dl.test_cases else 0

    print("\n" + "=" * 75)
    print("EVALUATION -- Basket Completion (complementary products)")
    print(f"  Models     : {len(models_dict)}")
    print(f"  Test cases : {n_cases:,}  (user-based 80/20 temporal split)")
    print(f"  Avg truth  : {avg_truth:.1f} SP/order")
    print(f"  K values   : {ks}")
    print(f"\n  [Ground truth] Tất cả SP còn lại trong cùng order")
    print(f"  [Cross-dept]   Ưu tiên model (qua bonus), không lọc trong eval")
    print("=" * 75)

    all_results = {}
    for name, fn in models_dict.items():
        all_results[name] = evaluate_one(fn, name, ks)

    df = pd.DataFrame(all_results).T
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    k_main = min(ks)
    h_col  = f'H@{k_main}'
    if h_col in df.columns:
        df.insert(0, 'Rank', df[h_col].rank(ascending=False).astype(int))
        df = df.sort_values('Rank')

    print(f"\n{'='*95}")
    print(f"  RESULTS  (n={n_cases:,} | avg_truth={avg_truth:.1f} SP/order)")
    print(f"{'='*95}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*95}")

    # Best per metric
    print("\n  BEST MODEL:")
    for k in ks:
        for prefix in ['H', 'NDCG']:
            col = f'{prefix}@{k}'
            if col in df.columns:
                best = df[col].idxmax()
                val  = df.loc[best, col]
                print(f"  {col:<10}: {best}  ({val:.4f})")

    # Improvement vs baseline
    names = list(models_dict.keys())
    if len(names) > 1:
        base = names[0]
        print(f"\n  IMPROVEMENT vs '{base}':")
        for model in names[1:]:
            parts = []
            for k in ks:
                col   = f'H@{k}'
                if col not in df.columns: continue
                new   = df.loc[model, col]
                old   = df.loc[base, col]
                delta = new - old
                pct   = (delta / old * 100) if old > 0 else 0
                sign  = "+" if delta >= 0 else ""
                parts.append(f"H@{k}:{sign}{delta:.4f}({sign}{pct:.1f}%)")
            print(f"  {model:<25}:  " + "  ".join(parts))

    # Benchmark guide
    print(f"""
  BENCHMARK GUIDE (với 50k SP, sparse 99.9%):
    H@10 < 0.10  = còn thấp -- cần cải thiện model/data
    H@10 ~ 0.15  = chấp nhận được
    H@10 ~ 0.25  = tốt
    H@10 > 0.35  = rất tốt (ngang production system)
    NDCG@10 > H@10 * 0.6 = model xếp hạng tốt (đúng ở top)
""")

    df.to_csv(PATH_OUTPUT_CSV, encoding='utf-8-sig')
    print(f"  Saved -> {PATH_OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    # === Chạy evaluation với temporal test cases ===
    from src.data_loader import load_temporal_test_cases

    # Tạo test cases (chạy 1 lần, tốn ~30s)
    dl.test_cases, model_df = load_temporal_test_cases(train_ratio=0.8)

    # Định nghĩa các rec_func từ ma trận similarity có sẵn
    from scipy.sparse import load_npz
    from src.config import MODELS_DIR, EVAL_KS

    def _make_rec_func(matrix, name):
        def rec_func(seed, top_k):
            if seed >= matrix.shape[0]:
                return []
            row = matrix[seed]
            if row.nnz == 0:
                return []
            order = np.argsort(row.data)[::-1]
            return row.indices[order[:top_k]].tolist()
        rec_func.__name__ = name
        return rec_func

    models = {}

    # # CB
    # try:
    #     m = load_npz(MODELS_DIR / "cb_similarity.npz")
    #     models["CB"] = _make_rec_func(m, "CB")
    # except Exception as e:
    #     print(f"  Skip CB: {e}")

    # SPMI
    try:
        m = load_npz(MODELS_DIR / "spmi_matrix.npz")
        models["SPMI"] = _make_rec_func(m, "SPMI")
    except Exception as e:
        print(f"  Skip SPMI: {e}")

    # # KG
    # try:
    #     m = load_npz(MODELS_DIR / "kg_similarity.npz")
    #     models["KG"] = _make_rec_func(m, "KG")
    # except Exception as e:
    #     print(f"  Skip KG: {e}")

    # # Hybrid
    # try:
    #     m = load_npz(MODELS_DIR / "hybrid_matrix.npz")
    #     models["Hybrid"] = _make_rec_func(m, "Hybrid")
    # except Exception as e:
    #     print(f"  Skip Hybrid: {e}")

    if not models:
        print("  [Eval] No models loaded. Nothing to evaluate.")
        sys.exit(1)

    ks = tuple(sorted(set(list(EVAL_KS) + [10, 50, 100])))
    run_comparison(models, ks=ks)