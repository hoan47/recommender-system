"""
test_cb.py — Script đánh giá kết quả Content-Based Similarity
File tạm, không cập nhật docs

Phân tích:
1. Thống kê mô tả similarity
2. Tỉ lệ cặp có similarity > 0
3. Phân tích word_overlap
4. Tương quan similarity vs overlap
5. Top 10 cặp tương đồng nhất
6. Phân tích similarity theo số từ
7. Nhận xét tổng quan
"""

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import pearsonr, spearmanr

# ─── Cấu hình ─────────────────────────────────────────────
FILE_PATH = "results/cb_similarity_distribution/cb_similarity_samples.csv"
OUTPUT_JSON = "results/cb_similarity_distribution/eval_summary.json"
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# ─── Đọc dữ liệu ──────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[+] Đọc file: {path}")
    df = pd.read_csv(path, encoding='utf-8')
    print(f"    → Số dòng: {len(df):,}")
    print(f"    → Các cột: {list(df.columns)}")
    return df

# ─── 1. Thống kê mô tả similarity ─────────────────────────
def stat_similarity(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("1. THỐNG KÊ MÔ TẢ similarity")
    print("=" * 55)

    s = df['similarity']
    desc = s.describe(percentiles=[0.25, 0.5, 0.75])
    result = {
        "count": int(desc['count']),
        "mean": round(desc['mean'], 6),
        "std": round(desc['std'], 6),
        "min": round(desc['min'], 6),
        "25%": round(desc['25%'], 6),
        "50%": round(desc['50%'], 6),
        "75%": round(desc['75%'], 6),
        "max": round(desc['max'], 6)
    }
    print(f"  Count     : {result['count']:>10,}")
    print(f"  Mean      : {result['mean']:>10.6f}")
    print(f"  Std       : {result['std']:>10.6f}")
    print(f"  Min       : {result['min']:>10.6f}")
    print(f"  25%       : {result['25%']:>10.6f}")
    print(f"  50%       : {result['50%']:>10.6f}")
    print(f"  75%       : {result['75%']:>10.6f}")
    print(f"  Max       : {result['max']:>10.6f}")
    return result

# ─── 2. Tỉ lệ similarity > 0 ──────────────────────────────
def ratio_nonzero(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("2. TỈ LỆ CẶP similarity > 0")
    print("=" * 55)

    total = len(df)
    pos = (df['similarity'] > 0).sum()
    zero = total - pos
    result = {
        "total": total,
        "similarity_gt_0": int(pos),
        "similarity_eq_0": int(zero),
        "ratio_gt_0": round(pos / total * 100, 4),
        "ratio_eq_0": round(zero / total * 100, 4)
    }
    print(f"  Tổng số cặp          : {result['total']:>10,}")
    print(f"  similarity > 0        : {result['similarity_gt_0']:>10,}  ({result['ratio_gt_0']:.2f}%)")
    print(f"  similarity = 0        : {result['similarity_eq_0']:>10,}  ({result['ratio_eq_0']:.2f}%)")
    print(f"\n  → Nhận xét: {result['ratio_gt_0']:.2f}% cặp có similarity dương.")
    if result['ratio_gt_0'] < 5:
        print("  → Rất thưa (sparse), đa số cặp sản phẩm không có từ chung.")
    elif result['ratio_gt_0'] < 20:
        print("  → Tỉ lệ thấp, nhiều cặp không có từ chung.")
    else:
        print("  → Tỉ lệ khá, nhiều cặp có điểm tương đồng.")
    return result

# ─── 3. Phân tích word_overlap ─────────────────────────────
def analyze_word_overlap(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("3. PHÂN TÍCH word_overlap")
    print("=" * 55)

    has_overlap = df[df['word_overlap'] > 0]
    no_overlap = df[df['word_overlap'] == 0]

    result = {
        "has_overlap_count": len(has_overlap),
        "has_overlap_ratio": round(len(has_overlap) / len(df) * 100, 4),
        "no_overlap_count": len(no_overlap),
        "no_overlap_ratio": round(len(no_overlap) / len(df) * 100, 4),
        "mean_similarity_when_overlap": round(has_overlap['similarity'].mean(), 6) if len(has_overlap) > 0 else 0
    }
    if len(has_overlap) > 0:
        result["max_similarity_when_overlap"] = round(has_overlap['similarity'].max(), 6)
        result["min_similarity_when_overlap"] = round(has_overlap['similarity'].min(), 6)
        result["std_similarity_when_overlap"] = round(has_overlap['similarity'].std(), 6)
    else:
        result["max_similarity_when_overlap"] = 0
        result["min_similarity_when_overlap"] = 0
        result["std_similarity_when_overlap"] = 0

    print(f"  Có word_overlap > 0 : {result['has_overlap_count']:>10,}  ({result['has_overlap_ratio']:.2f}%)")
    print(f"  Word_overlap = 0    : {result['no_overlap_count']:>10,}  ({result['no_overlap_ratio']:.2f}%)")
    print(f"\n  -- Với các cặp có word_overlap > 0 --")
    print(f"  Mean similarity     : {result['mean_similarity_when_overlap']:.6f}")
    print(f"  Min similarity      : {result['min_similarity_when_overlap']:.6f}")
    print(f"  Max similarity      : {result['max_similarity_when_overlap']:.6f}")
    print(f"  Std similarity      : {result['std_similarity_when_overlap']:.6f}")
    return result

# ─── 4. Tương quan similarity vs overlap ──────────────────
def correlation_sim_overlap(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("4. TƯƠNG QUAN similarity vs overlap")
    print("=" * 55)

    # Lọc các cặp có ít nhất một giá trị dương để tránh bias từ zero
    mask = (df['similarity'] > 0) | (df['overlap'] > 0)
    sub = df[mask]
    print(f"  Số cặp có similarity>0 hoặc overlap>0: {len(sub):,} / {len(df):,}")

    if len(sub) < 2:
        print("  → Không đủ dữ liệu để tính tương quan.")
        return {"pearson_r": 0, "pearson_p": 0, "spearman_r": 0, "spearman_p": 0, "n_used": len(sub)}

    corr_pearson, p_pearson = pearsonr(sub['similarity'], sub['overlap'])
    corr_spearman, p_spearman = spearmanr(sub['similarity'], sub['overlap'])

    result = {
        "pearson_r": round(corr_pearson, 6),
        "pearson_p": round(p_pearson, 6),
        "spearman_r": round(corr_spearman, 6),
        "spearman_p": round(p_spearman, 6),
        "n_used": len(sub)
    }
    print(f"  Pearson r  = {result['pearson_r']:.6f}  (p = {result['pearson_p']:.6e})")
    print(f"  Spearman ρ = {result['spearman_r']:.6f}  (p = {result['spearman_p']:.6e})")
    if abs(result['pearson_r']) > 0.7:
        print("  → Tương quan tuyến tính mạnh giữa similarity TF-IDF và overlap.")
    elif abs(result['pearson_r']) > 0.4:
        print("  → Tương quan trung bình.")
    else:
        print("  → Tương quan yếu, hai metric đo lường khác nhau.")
    if abs(result['spearman_r']) > 0.7:
        print("  → Tương quan rank mạnh.")
    elif abs(result['spearman_r']) > 0.4:
        print("  → Tương quan rank trung bình.")
    else:
        print("  → Tương quan rank yếu.")
    return result

# ─── 5. Top 10 cặp tương đồng nhất ────────────────────────
def top_n_similar(df: pd.DataFrame, n: int = 10) -> list:
    print(f"\n" + "=" * 55)
    print(f"5. TOP {n} CẶP CÓ similarity CAO NHẤT")
    print("=" * 55)

    top = df.nlargest(n, 'similarity')
    records = []
    print(f"  {'#':>3}  {'prod_a':>7}  {'prod_b':>7}  {'similarity':>12}  {'word_overlap':>7}  {'overlap':>8}")
    print(f"  {'---':>3}  {'-------':>7}  {'-------':>7}  {'-----------':>12}  {'----------':>7}  {'-------':>8}")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        rec = {
            "rank": i,
            "product_a": int(row['product_a_id']),
            "product_b": int(row['product_b_id']),
            "similarity": round(row['similarity'], 6),
            "word_overlap": int(row['word_overlap']),
            "overlap": round(row['overlap'], 6)
        }
        records.append(rec)
        print(f"  {i:>3}  {rec['product_a']:>7}  {rec['product_b']:>7}  {rec['similarity']:>12.6f}  {rec['word_overlap']:>7}  {rec['overlap']:>8.6f}")
    return records

# ─── 6. Phân tích similarity theo số từ ────────────────────
def analyze_by_word_count(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("6. PHÂN TÍCH similarity THEO SỐ LƯỢNG TỪ")
    print("=" * 55)

    # Lấy số từ trung bình của mỗi cặp
    df['avg_word_count'] = (df['word_count_a'] + df['word_count_b']) / 2.0

    # Chia nhóm
    bins = [0, 2, 4, 6, 8, 10, 20, 50, 100]
    labels = ['1-2', '3-4', '5-6', '7-8', '9-10', '11-20', '21-50', '51+']
    df['word_bin'] = pd.cut(df['avg_word_count'], bins=bins, labels=labels, right=True)

    grouped = df.groupby('word_bin', observed=True)['similarity'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(f"  {'Nhóm từ':>8}  {'Số cặp':>8}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'--------':>8}  {'------':>8}  {'------':>10}  {'-----':>10}  {'-----':>10}  {'-----':>10}")
    result = {}
    for label, row in grouped.iterrows():
        result[str(label)] = {
            "count": int(row['count']),
            "mean_similarity": round(row['mean'], 6),
            "std_similarity": round(row['std'], 6),
            "min_similarity": round(row['min'], 6),
            "max_similarity": round(row['max'], 6)
        }
        print(f"  {label:>8}  {int(row['count']):>8,}  {row['mean']:>10.6f}  {row['std']:>10.6f}  {row['min']:>10.6f}  {row['max']:>10.6f}")

    # Xu hướng
    means = grouped['mean'].dropna()
    if len(means) > 1:
        trend = "giảm" if means.iloc[0] > means.iloc[-1] else "tăng"
        print(f"\n  → Xu hướng: similarity có xu hướng {trend} khi số từ tăng.")
    else:
        print("\n  → Không đủ nhóm để nhận xét xu hướng.")
    return result

# ─── 7. Nhận xét tổng quan ────────────────────────────────
def summarize(stat_sim: dict, ratio: dict, overlap: dict, corr: dict) -> str:
    print("\n" + "=" * 55)
    print("7. NHẬN XÉT TỔNG QUAN")
    print("=" * 55)

    lines = []
    lines.append(f"- Tổng số cặp sản phẩm: {stat_sim['count']:,}")
    lines.append(f"- Similarity trung bình: {stat_sim['mean']:.6f} (rất thấp, do đa số = 0)")
    lines.append(f"- Tỉ lệ cặp có similarity > 0: {ratio['ratio_gt_0']:.2f}% ({ratio['similarity_gt_0']:,} cặp)")

    # Đánh giá mức độ thưa
    if ratio['ratio_gt_0'] < 5:
        lines.append("- Mức độ thưa: RẤT CAO — ma trận similarity gần như sparse (>95% bằng 0).")
    elif ratio['ratio_gt_0'] < 15:
        lines.append("- Mức độ thưa: CAO.")
    elif ratio['ratio_gt_0'] < 30:
        lines.append("- Mức độ thưa: TRUNG BÌNH.")
    else:
        lines.append("- Mức độ thưa: THẤP — nhiều cặp có điểm tương đồng.")

    # Word overlap
    lines.append(f"- Tỉ lệ cặp có từ chung (word_overlap > 0): {overlap['has_overlap_ratio']:.2f}%")
    if overlap['has_overlap_count'] > 0:
        lines.append(f"- Trung bình similarity khi có từ chung: {overlap['mean_similarity_when_overlap']:.6f}")

    # Tương quan
    if corr['n_used'] >= 2:
        lines.append(f"- Tương quan Pearson(sim, overlap): r = {corr['pearson_r']:.4f}")
        lines.append(f"- Tương quan Spearman(sim, overlap): ρ = {corr['spearman_r']:.4f}")

    # Kết luận
    lines.append("")
    lines.append("=== KẾT LUẬN ===")
    if ratio['ratio_gt_0'] < 5:
        lines.append("CB similarity hiện tại rất thưa. Cần kiểm tra:")
        lines.append("  1. Dữ liệu mô tả sản phẩm: có quá ngắn / thiếu đặc trưng?")
        lines.append("  2. Vectorizer: có dùng đúng TF-IDF với unigram/bigram?")
        lines.append("  3. Stopwords: có filter quá nhiều từ?")
        lines.append("  4. Có cần mở rộng đặc trưng (category, brand...)?")

        # Gợi ý cải thiện
        lines.append("")
        lines.append("Gợi ý cải thiện:")
        lines.append("  - Kiểm tra product_filter.py có đang filter đúng không")
        lines.append("  - Thử dùng CountVectorizer thay vì TfidfVectorizer")
        lines.append("  - Bổ sung thêm cột đặc trưng: department, aisle, brand")
    elif ratio['ratio_gt_0'] < 15:
        lines.append("CB similarity tương đối thưa. Có thể cải thiện bằng cách bổ sung đặc trưng.")
    else:
        lines.append("CB similarity hoạt động tương đối tốt.")

    report = "\n".join(lines)
    print(report)
    return report

# ─── Main ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ĐÁNH GIÁ Content-Based Similarity")
    print("  File: cb_similarity_samples.csv")
    print("=" * 55)

    df = load_data(FILE_PATH)

    # 1
    stat_sim = stat_similarity(df)

    # 2
    ratio = ratio_nonzero(df)

    # 3
    overlap = analyze_word_overlap(df)

    # 4
    corr = correlation_sim_overlap(df)

    # 5
    top10 = top_n_similar(df, n=10)

    # 6
    wc_analysis = analyze_by_word_count(df)

    # 7
    summary = summarize(stat_sim, ratio, overlap, corr)

    # ─── Ghi JSON ──────────────────────────────────────
    output = {
        "file": FILE_PATH,
        "thong_ke_similarity": stat_sim,
        "ti_le_similarity_gt_0": ratio,
        "phan_tich_word_overlap": overlap,
        "tuong_quan_sim_overlap": corr,
        "top_10_cap_tuong_dong": top10,
        "phan_tich_theo_so_tu": wc_analysis,
        "nhan_xet": summary
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Đã lưu báo cáo JSON: {OUTPUT_JSON}")

    print("\n" + "=" * 55)
    print("  HOÀN TẤT ĐÁNH GIÁ")
    print("=" * 55)

if __name__ == "__main__":
    main()