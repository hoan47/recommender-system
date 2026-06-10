# Bảng Theo Dõi Tiến Độ (Progress Tracking)

Cập nhật lần cuối: 2026-06-10 18:36

---

## Hiện trạng file source code

| File | Trạng thái | Ghi chú |
|------|-----------|---------|
| `src/config.py` | ✅ Hoàn tất | Hyperparameters, paths + Product Filter Strategy |
| `src/features/loader.py` | ✅ Hoàn tất | Load CSV (chunksize=500K), merge aisle/department |
| `src/features/product_filter.py` | ✅ Hoàn tất | Bộ lọc non-food departments/aisles khỏi train data |
| `src/features/vectorizer.py` | ✅ Hoàn tất | TF-IDF + one-hot, cosine similarity on-demand |
| `src/utils/__init__.py` | ✅ Hoàn tất | Package init |
| `src/utils/_numba_ops.py` | ✅ Hoàn tất | Numba-accelerated ops (counting, adjacency CSR) |
| `src/models/cb_filter.py` | ✅ Hoàn tất | CB Diversity Filter, threshold=0.3 |
| `src/models/ochiai.py` | ✅ Hoàn tất | Ochiai + Confidence Score, CSR sparse matrix — dùng Numba counting |
| `src/models/item2vec.py` | ✅ Hoàn tất | Word2Vec Skip-gram (gensim) |
| `src/models/deepwalk.py` | ✅ Hoàn tất | Graph embedding + uniform random walk — dùng Numba counting + adjacency CSR |
| `src/models/assoc_rules.py` | ✅ Hoàn tất | Association Rules từ co-occurrence matrix |
| `src/models/ensemble.py` | ✅ Hoàn tất | Weighted ensemble (α=0.5, β=0.25, γ=0.25) + CB Filter |
| `src/evaluation/__init__.py` | ✅ Hoàn tất | Package init (hiện rỗng) |

## Scripts chạy từng bước

| Script | Trạng thái | Ghi chú |
|--------|-----------|---------|
| `scripts/01_load_data.py` | ✅ Hoàn tất | Load + cache parquet + Product Filter Strategy |
| `scripts/02_cb_filter.py` | ✅ Hoàn tất | Train CB Filter |
| `scripts/03_ochiai.py` | ✅ Hoàn tất | Train Ochiai |
| `scripts/04_item2vec.py` | ✅ Hoàn tất | Train Item2Vec |
| `scripts/05_deepwalk.py` | ✅ Hoàn tất | Train DeepWalk |
| `scripts/06_assoc_rules.py` | ✅ Hoàn tất | Train Association Rules + test recommend |
| `scripts/07_ensemble.py` | ✅ Hoàn tất | Ensemble + test recommend |
| `scripts/08_streamlit_app.py` | ✅ Hoàn tất | Streamlit Dashboard UI |

## Model đã train (cached)

| Model | File | Kích thước | Đã train? |
|-------|------|-----------|----------|
| CB Filter | `models/cb_filter/product_vectors.npz` | ~35 MB | ✅ Có |
| Ochiai | `models/ochiai/cooc_matrix.npz` | ~28 MB | ✅ Có (metadata.json đã fix) |
| Item2Vec | `models/item2vec/` | - | ✅ Có |
| DeepWalk | `models/deepwalk/` | - | ✅ Có |
| Assoc Rules | `models/assoc_rules/` | - | ✅ Có |

## Product Filter Strategy (mới thêm)

| Cấu hình | Giá trị | Ghi chú |
|----------|--------|---------|
| `EXCLUDED_DEPARTMENTS` | `[8, 11, 17, 2, 21]` | pets, personal care, household, other, missing |
| `EXCLUDED_AISLES` | `[82, 102]` | baby accessories, baby bath body care (babies dept) |
| Aisle giữ lại | `92` | baby food formula |

- Filter áp dụng trong `scripts/01_load_data.py` — loại non-food products khỏi `order_products.parquet`
- `products.parquet` **không** bị lọc (CB Filter cần toàn bộ products để vectorize)
- Các scripts model (02-07) không cần sửa vì đọc parquet đã lọc sẵn

## Còn lại

- [x] Chạy `scripts/04_item2vec.py` — train Item2Vec
- [x] Chạy `scripts/05_deepwalk.py` — train DeepWalk
- [x] Chạy `scripts/06_assoc_rules.py` — train Assoc Rules
- [x] Chạy `scripts/07_ensemble.py` — test ensemble + CB Filter
- [x] Tạo `scripts/08_streamlit_app.py` — Dashboard trực quan
- [ ] Chạy `streamlit run scripts/08_streamlit_app.py` — mở Dashboard
- [x] ~~CB Filter đang không lọc được (threshold 0.8 quá cao) — cần debug~~
- [x] ~~Tối ưu threshold CB Filter~~ ✅ CB_THRESHOLD=0.3 đã hoạt động
- [ ] Evaluation pipeline (survey + LLM + metrics)
- [ ] Đánh giá model qua survey
- [ ] Chạy `streamlit run scripts/08_streamlit_app.py` — mở Dashboard