# evaluate.py
# Danh gia chinh xac cho bai toan basket completion / complementary recommendation
#
# THIET KE DUNG:
#   - Ground truth = TAT CA SP con lai trong cung order (khong phan biet dept)
#   - Ly do: model goi y "mua kem" -> nguoi dung muon mua them, bat ky dept nao
#   - Trong thuc te, mua them dau, rau, nuoc ngot voi ga dong lanh deu la "dung"
#   - Cross-dept uu tien la viec cua MODEL (qua cross_bonus), khong phai cua EVALUATOR
#
# TEST SPLIT:
#   - User-based: 80% order dau lam train, 20% order cuoi lam test
#   - Dam bao khong co data leakage
#   - Moi test order co ~8-12 SP -> nhieu co hoi hit
#
# METRICS DUNG:
#   - H@K (Hit Rate): ti le order co >= 1 goi y dung -> metric chinh
#   - P@K, R@K, F1@K: bo sung
#   - Khong dung MAE/MSE/RMSE (khong phu hop voi ranking task)
#
# BENCHMARK THUC TE:
#   Voi dataset 50k SP, sparse 99.9%:
#     H@10 < 0.10  = con thap
#     H@10 ~ 0.15  = chap nhan duoc
#     H@10 ~ 0.25  = tot
#     H@10 > 0.35  = rat tot (ngang Amazon/Instacart production)

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import data_loader as dl
from config import PATH_OUTPUT_CSV


def calc_metrics(recommended: list,
                 ground_truth: list,
                 ks: tuple = (10, 50, 100)) -> dict:
    if not ground_truth or not recommended:
        return {}

    truth_set   = set(ground_truth)
    total_truth = len(truth_set)
    results     = {}

    for k in ks:
        top_k      = recommended[:int(k)]
        hits       = len(set(top_k) & truth_set)
        precision  = hits / int(k)
        recall     = hits / total_truth
        f1         = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        hit_rate   = 1.0 if hits > 0 else 0.0
        ndcg       = _ndcg(top_k, truth_set, int(k))

        results[f'H@{int(k)}']    = hit_rate
        results[f'NDCG@{int(k)}'] = ndcg
        results[f'P@{int(k)}']    = precision
        results[f'R@{int(k)}']    = recall
        results[f'F1@{int(k)}']   = f1

    return results


def _ndcg(top_k: list, truth_set: set, k: int) -> float:
    """NDCG@K: tinh den vi tri xep hang cua hit."""
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top_k) if p in truth_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_one(rec_func, name: str,
                 ks: tuple = (10, 50, 100)) -> dict:
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
            for p in ['H', 'NDCG', 'P', 'R', 'F1']:
                r[f'{p}@{k}'] = 0.0
        return r

    result = {key: round(float(np.mean(vals)), 4)
              for key, vals in metric_lists.items()}
    result['Time(s)']  = elapsed
    result['N_valid']  = len(next(iter(metric_lists.values())))
    return result


def run_comparison(models_dict: dict,
                   ks: tuple = (10, 50, 100)) -> pd.DataFrame:

    # Uu tien H@K va NDCG@K trong col_order (metrics quan trong nhat)
    col_order = []
    for k in ks:
        k_int = int(k)
        col_order.extend([f'H@{k_int}', f'NDCG@{k_int}', f'P@{k_int}', f'R@{k_int}', f'F1@{k_int}'])
    col_order += ['Time(s)', 'N_valid']

    # Thong ke test set
    n_cases  = len(dl.test_cases)
    avg_truth = np.mean([len(t) for _, t in dl.test_cases]) if dl.test_cases else 0

    print("\n" + "=" * 75)
    print("EVALUATION -- Basket Completion (complementary products)")
    print(f"  Models     : {len(models_dict)}")
    print(f"  Test cases : {n_cases:,}  (user-based 20% split)")
    print(f"  Avg truth  : {avg_truth:.1f} SP/order")
    print(f"  K values   : {ks}")
    print(f"\n  [Ground truth] Tat ca SP con lai trong cung order")
    print(f"  [Cross-dept]   Uu tien model (qua bonus), khong loc trong eval")
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
            print(f"  {model:<25}: " + "  ".join(parts))

    # Benchmark guide
    print(f"""
  BENCHMARK GUIDE (voi 50k SP, sparse 99.9%):
    H@10 < 0.10  = con thap -- can cai thien model/data
    H@10 ~ 0.15  = chap nhan duoc
    H@10 ~ 0.25  = tot
    H@10 > 0.35  = rat tot (ngang production system)
    NDCG@10 > H@10 * 0.6 = model xep hang tot (dung o top)
""")

    df.to_csv(PATH_OUTPUT_CSV, encoding='utf-8-sig')
    print(f"  Saved -> {PATH_OUTPUT_CSV}")
    return df