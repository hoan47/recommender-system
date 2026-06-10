# Bảng Theo Dõi Tiến Độ (Progress Tracking)

Cập nhật lần cuối: 2026-06-10 02:30

---

## Hiện trạng file source code

| File | Trạng thái | Ghi chú |
|------|-----------|---------|
| `src/config.py` | ✅ Hoàn tất | Hyperparameters, paths |
| `src/features/loader.py` | ✅ Hoàn tất | Load CSV (chunksize=500K), merge aisle/department |
| `src/features/vectorizer.py` | ✅ Hoàn tất | TF-IDF + one-hot, cosine similarity on-demand |
| `src/utils/__init__.py` | ✅ Hoàn tất | Package init |
| `src/utils/_numba_ops.py` | ✅ Hoàn tất | Numba-accelerated ops (counting, adjacency CSR) |
| `src/models/cb_filter.py` | ✅ Hoàn tất | CB Diversity Filter, threshold=0.8 |
| `src/models/ochiai.py` | ✅ Hoàn tất | Ochiai + Confidence Score, CSR sparse matrix — dùng Numba counting |
| `src/models/item2vec.py` | ✅ Hoàn tất | Word2Vec Skip-gram (gensim) |
| `src/models/deepwalk.py` | ✅ Hoàn tất | Graph embedding + uniform random walk — dùng Numba counting + adjacency CSR |
| `src/models/assoc_rules.py` | ✅ Hoàn tất | Association Rules từ co-occurrence matrix |
| `src/models/ensemble.py` | ✅ Hoàn tất | Weighted ensemble (α=0.5, β=0.25, γ=0.25) + CB Filter |
| `src/evaluation/__init__.py` | ✅ Hoàn tất | Package init (hiện rỗng) |

## Scripts chạy từng bước

| Script | Trạng thái | Ghi chú |
|--------|-----------|---------|
| `scripts/01_load_data.py` | ✅ Hoàn tất | Load + cache parquet |
| `scripts/02_cb_filter.py` | ✅ Hoàn tất | Train CB Filter |
| `scripts/03_ochiai.py` | ✅ Hoàn tất | Train Ochiai |
| `scripts/04_item2vec.py` | ✅ Hoàn tất | Train Item2Vec |
| `scripts/05_deepwalk.py` | ✅ Hoàn tất | Train DeepWalk |
| `scripts/06_assoc_rules.py` | ✅ Hoàn tất | Train Association Rules + test recommend |
| `scripts/07_ensemble.py` | ✅ Hoàn tất | Ensemble + test recommend |

## Model đã train (cached)

| Model | File | Kích thước | Đã train? |
|-------|------|-----------|----------|
| CB Filter | `models/cb_filter/product_vectors.npz` | ~35 MB | ✅ Có |
| Ochiai | `models/ochiai/cooc_matrix.npz` | ~28 MB | ✅ Có (metadata.json đã fix) |
| Item2Vec | `models/item2vec/` | - | ❌ Chưa |
| DeepWalk | `models/deepwalk/` | - | ❌ Chưa |
| Assoc Rules | `models/assoc_rules/` | - | ❌ Chưa |

## Còn lại

- [ ] Chạy `scripts/04_item2vec.py` — train Item2Vec (mất ~5-10 phút)
- [ ] Chạy `scripts/05_deepwalk.py` — train DeepWalk (mất ~3-5 phút)
- [ ] Chạy `scripts/06_assoc_rules.py` — train Assoc Rules (mất ~2-5 phút)
- [ ] Chạy `scripts/07_ensemble.py` — test ensemble + CB Filter