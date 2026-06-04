import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import data_loader as dl
from config import PATH_OUTPUT_CSV


def calc_metrics(
    recommended: list,
    ground_truth: list,
    ks: tuple = (10, 50, 100),
) -> dict:
    if not ground_truth or not recommended:
        return {}

    truth_set   = set(ground_truth)
    total_truth = len(truth_set)
    results     = {}

    for k in ks:
        top_k     = recommended[:k]
        hits      = len(set(top_k) & truth_set)
        precision = hits / k
        recall    = hits / total_truth
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results[f"H@{k}"]    = 1.0 if hits > 0 else 0.0
        results[f"NDCG@{k}"] = _ndcg(top_k, truth_set, k)
        results[f"P@{k}"]    = precision
        results[f"R@{k}"]    = recall
        results[f"F1@{k}"]   = f1

    return results


def _ndcg(top_k: list, truth_set: set, k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top_k) if p in truth_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_one(rec_func, name: str, ks: tuple = (10, 50, 100)) -> dict:
    metric_lists: dict = defaultdict(list)
    t0 = time.time()

    for seed, truth in tqdm(dl.test_cases, desc=f"  [{name}]", ncols=80):
        if not truth:
            continue
        try:
            recs = rec_func(seed, max(ks))
            if not recs:
                continue
            for key, val in calc_metrics(recs, truth, ks).items():
                metric_lists[key].append(val)
        except Exception:
            continue

    elapsed = round(time.time() - t0, 2)
    if not metric_lists:
        result = {"Time(s)": elapsed}
        for k in ks:
            for p in ["H", "NDCG", "P", "R", "F1"]:
                result[f"{p}@{k}"] = 0.0
        return result

    result = {key: round(float(np.mean(vals)), 4) for key, vals in metric_lists.items()}
    result["Time(s)"] = elapsed
    result["N_valid"] = len(next(iter(metric_lists.values())))
    return result


def run_comparison(models_dict: dict, ks: tuple = (10, 50, 100)) -> pd.DataFrame:
    col_order = []
    for k in ks:
        col_order.extend([f"H@{k}", f"NDCG@{k}", f"P@{k}", f"R@{k}", f"F1@{k}"])
    col_order += ["Time(s)", "N_valid"]

    n_cases   = len(dl.test_cases)
    avg_truth = np.mean([len(t) for _, t in dl.test_cases]) if dl.test_cases else 0

    print("\n" + "=" * 75)
    print("EVALUATION — Basket Completion")
    print(f"  Models     : {len(models_dict)}")
    print(f"  Test cases : {n_cases:,}")
    print(f"  Avg truth  : {avg_truth:.1f} items/order")
    print(f"  K values   : {ks}")
    print("=" * 75)

    all_results = {name: evaluate_one(fn, name, ks) for name, fn in models_dict.items()}

    df        = pd.DataFrame(all_results).T
    col_order = [c for c in col_order if c in df.columns]
    df        = df[col_order]

    k_main = min(ks)
    h_col  = f"H@{k_main}"
    if h_col in df.columns:
        df.insert(0, "Rank", df[h_col].rank(ascending=False).astype(int))
        df = df.sort_values("Rank")

    print(f"\n{'='*95}")
    print(f"  RESULTS  (n={n_cases:,} | avg_truth={avg_truth:.1f})")
    print(f"{'='*95}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*95}")

    print("\n  BEST MODEL PER METRIC:")
    for k in ks:
        for prefix in ["H", "NDCG"]:
            col = f"{prefix}@{k}"
            if col in df.columns:
                best = df[col].idxmax()
                val  = df.loc[best, col]
                print(f"  {col:<10}: {best}  ({val:.4f})")

    names = list(models_dict.keys())
    if len(names) > 1:
        base = names[0]
        print(f"\n  IMPROVEMENT vs '{base}':")
        for model in names[1:]:
            parts = []
            for k in ks:
                col = f"H@{k}"
                if col not in df.columns:
                    continue
                delta = df.loc[model, col] - df.loc[base, col]
                pct   = (delta / df.loc[base, col] * 100) if df.loc[base, col] > 0 else 0
                sign  = "+" if delta >= 0 else ""
                parts.append(f"H@{k}:{sign}{delta:.4f}({sign}{pct:.1f}%)")
            print(f"  {model:<25}: " + "  ".join(parts))

    df.to_csv(PATH_OUTPUT_CSV, encoding="utf-8-sig")
    print(f"\n  Saved -> {PATH_OUTPUT_CSV}")
    return df
