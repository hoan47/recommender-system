"""
SPMI — Shifted Positive PMI từ co-occurrence
Tìm sản phẩm MUA KÈM (complementary)
"""
import gc, math, json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from tqdm import tqdm

from src.config import MODELS_DIR, SPMI_K, TOTAL_PRIOR_ORDERS, SPMI_TOP_K

COOC_FILE = MODELS_DIR / "cooc_matrix.npz"
SPMI_FILE = MODELS_DIR / "spmi_matrix.npz"

def build_cooc(prior_df):
    """
    Đếm co-occurrence từ prior orders
    Trả về (cooc_csr, order_freqs) — sparse + numpy array
    """
    print("\n  [SPMI] Building co-occurrence ...")
    n = prior_df["product_id"].max() + 1
    grouped = prior_df.groupby("order_id")
    cooc = lil_matrix((n, n), dtype=np.float64)
    freq = np.zeros(n, dtype=np.float64)

    for _, grp in tqdm(grouped, desc="  Co-occurrence"):
        prods = grp["product_id"].values
        if len(prods) < 2:
            continue
        uniq = np.unique(prods)
        freq[uniq] += 1
        for i in range(len(prods)):
            for j in range(i + 1, len(prods)):
                a, b = int(prods[i]), int(prods[j])
                cooc[a, b] += 1
                cooc[b, a] += 1

    cooc_csr = cooc.tocsr()
    del cooc; gc.collect()
    print(f"  [SPMI] Co-occurrence: {cooc_csr.nnz:,} entries")
    return cooc_csr, freq

def build_spmi(cooc_csr, freq, k=SPMI_K, top_k=SPMI_TOP_K):
    """
    Tính SPMI từ co-occurrence, chỉ giữ top-K mỗi dòng
    SPMI = max(log(cooc * N / (freq_i * freq_j)) - log(k), 0)
    """
    print(f"\n  [SPMI] Computing SPMI k={k} ...")
    n = cooc_csr.shape[0]
    log_shift = math.log(k)
    rows, cols, vals = [], [], []

    for i in tqdm(range(n), desc="  SPMI"):
        row = cooc_csr[i]
        if row.nnz == 0:
            continue
        fi = freq[i]
        if fi == 0:
            continue
        scores = []
        for j, c in zip(row.indices, row.data):
            fj = freq[j]
            if fj == 0:
                continue
            pmi = math.log(c * TOTAL_PRIOR_ORDERS / (fi * fj))
            spmi = max(pmi - log_shift, 0)
            if spmi > 0:
                scores.append((j, spmi))
        if not scores:
            continue
        # Giữ top-K
        scores.sort(key=lambda x: -x[1])
        for j, s in scores[:top_k]:
            rows.append(i)
            cols.append(j)
            vals.append(s)

    spmi = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    del rows, cols, vals; gc.collect()
    print(f"  [SPMI] Done: {spmi.nnz:,} entries")
    return spmi

def save(cooc, spmi):
    save_npz(COOC_FILE, cooc)
    save_npz(SPMI_FILE, spmi)
    print(f"  [SPMI] Saved: {COOC_FILE}, {SPMI_FILE}")

def load():
    return load_npz(COOC_FILE), load_npz(SPMI_FILE)

if __name__ == "__main__":
    from src.data_loader import load_prior
    prior = load_prior()
    cooc, freq = build_cooc(prior)
    spmi = build_spmi(cooc, freq)
    save(cooc, spmi)