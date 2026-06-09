# Bảng Theo Dõi Tiến Độ (Progress Tracking)

Cập nhật lần cuối: 2026-06-10 02:30

---

## Hiện trạng file source code

| File | Trạng thái | Ghi chú |
|------|-----------|---------|
| `src/config.py` | ✅ Hoàn tất | Hyperparameters, paths |
| `src/features/loader.py` | ✅ Hoàn tất | Load CSV (chunksize=500K), merge aisle/department |
| `src/features/vectorizer.py` | ✅ Hoàn tất | TF-IDF + one-hot, cosine similarity on-demand |
| `src/models/cb_filter.py` | ✅ Hoàn tất | CB Diversity Filter, threshold=0.8 |
| `src/models/ochiai.py` | ✅ Hoàn tất | Ochiai + Confidence Score, CSR sparse matrix |
| `src/models/item2vec.py` | ✅ Hoàn tất | Word2Vec Skip-gram (gensim) |
| `src/models/node2vec.py` | ✅ Hoàn tất | Graph embedding + random walk p/q |
| `src/models/assoc_rules.py` | ✅ Hoàn tất | Association Rules từ co-occurrence matrix |
| `src/models/ensemble.py` | ✅ Hoàn tất | Weighted ensemble (α=0.5, β=0.25, γ=0.25) + CB Filter |
| `src/evaluation/__init__.py` | ✅ Hoàn tất | Package init |
| `src/evaluation/metrics.py` | ✅ Hoàn tất | Precision@K, Recall@K, F1@K, Hit@K |
| `src/evaluation/survey_generator.py` | ✅ Hoàn tất | Sinh mẫu khảo sát (top5 + noise) |

## Scripts chạy từng bước

| Script | Trạng thái | Ghi chú |
|--------|-----------|---------|
| `scripts/01_load_data.py` | ✅ Hoàn tất | Load + cache parquet |
| `scripts/02_cb_filter.py` | ✅ Hoàn tất | Train CB Filter |
| `scripts/03_ochiai.py` | ✅ Hoàn tất | Train Ochiai |
| `scripts/04_item2vec.py` | ✅ Hoàn tất | Train Item2Vec |
| `scripts/05_node2vec.py` | ✅ Hoàn tất | Train Node2Vec |
| `scripts/06_assoc_rules.py` | ✅ Hoàn tất | Train Association Rules + test recommend |
| `scripts/07_ensemble.py` | ✅ Hoàn tất | Ensemble + test recommend |
| `scripts/08_evaluate_assoc_rules.py` | ✅ Hoàn tất | Đánh giá Assoc Rules: survey + metrics |

## Model đã train (cached)

| Model | File | Kích thước | Đã train? |
|-------|------|-----------|----------|
| CB Filter | `models/cb_filter/product_vectors.npz` | ~35 MB | ✅ Có |
| Ochiai | `models/ochiai/cooc_matrix.npz` | ~28 MB | ✅ Có (metadata.json đã fix) |
| Item2Vec | `models/item2vec/` | - | ❌ Chưa |
| Node2Vec | `models/node2vec/` | - | ❌ Chưa |
| Assoc Rules | `models/assoc_rules/` | - | ❌ Chưa |

## Còn lại

- [ ] Chạy `scripts/04_item2vec.py` — train Item2Vec (mất ~5-10 phút)
- [ ] Chạy `scripts/05_node2vec.py` — train Node2Vec (mất ~15-30 phút)
- [ ] Chạy `scripts/06_assoc_rules.py` — train Assoc Rules (mất ~2-5 phút)
- [ ] Chạy `scripts/07_ensemble.py` — test ensemble + CB Filter
- [ ] Chạy `scripts/08_evaluate_assoc_rules.py` — đánh giá Assoc Rules (khi đã train xong)
