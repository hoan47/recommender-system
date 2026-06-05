# 📊 Tiến độ dự án — Recommender System (Global / Item-Oriented)

Cập nhật lần cuối: 06/05/2026 22:36 (Asia/Saigon)

## ✅ Checklist hoàn thành

- [x] Tạo cấu trúc thư mục `src/` + `__init__.py` (utils, features, evaluation)
- [x] Tạo thư mục `models/`, `results/` (nếu chưa có)
- [x] `src/utils/data_loader.py` — load được tất cả dữ liệu, tách train/test từ orders.csv
- [x] `src/features/build_tfidf.py` — TF-IDF + cosine similarity (sparse)
- [x] `src/features/build_spmi.py` — co-occurrence → PMI → SPMI → tune k trên train → save
- [x] `src/features/build_knowledge_graph.py` — graph từ SPMI edges → node2vec → tune trên train → save
- [x] `src/features/build_hybrid.py` — normalize + hybrid score → grid search trên train → save weights
- [x] `src/evaluation/evaluate.py` — Recall@K, NDCG@K, MAP@K cho tất cả models (dùng test 1 lần)
- [ ] `results/metrics.json` — so sánh tất cả models (sẽ có sau khi chạy `evaluate.py`)

## 📁 Hiện trạng file

| File | Trạng thái | Mô tả |
|------|-----------|-------|
| `src/__init__.py` | ✅ Đã tạo | Package marker (trống) |
| `src/utils/__init__.py` | ✅ Đã tạo | Package marker (trống) |
| `src/utils/data_loader.py` | ✅ Đã tạo | ~170 dòng, load 5 file CSV, tách train/test |
| `src/features/__init__.py` | ✅ Đã tạo | Package marker (trống) |
| `src/features/build_tfidf.py` | ✅ Đã tạo | ~250 dòng, TF-IDF + cosine similarity |
| `src/features/build_spmi.py` | ✅ Đã tạo | ~370 dòng, co-occurrence → SPMI → tune k |
| `src/features/build_knowledge_graph.py` | ✅ Đã tạo | ~430 dòng, graph → node2vec → tune params |
| `src/features/build_hybrid.py` | ✅ Đã tạo | ~430 dòng, normalize → grid search α/β/threshold |
| `src/evaluation/__init__.py` | ✅ Đã tạo | Package marker (trống) |
| `src/evaluation/evaluate.py` | ✅ Đã tạo | ~370 dòng, Recall@K, NDCG@K, MAP@K |
| `docs/progress.md` | ✅ Đã tạo | File này |

## 📦 Output models (chưa chạy)

Tất cả file dưới đây sẽ được tạo khi chạy các script tương ứng:

| File | Script tạo |
|------|------------|
| `models/tfidf_matrix.npz` | `build_tfidf.py` |
| `models/item_similarity_cb.npz` | `build_tfidf.py` |
| `models/tfidf_vectorizer.pkl` | `build_tfidf.py` |
| `models/cooc_matrix.npz` | `build_spmi.py` |
| `models/spmi_matrix.npz` | `build_spmi.py` |
| `models/spmi_best_k.json` | `build_spmi.py` |
| `models/kg_embeddings.npy` | `build_knowledge_graph.py` |
| `models/kg_best_params.json` | `build_knowledge_graph.py` |
| `models/kg_similarity.npz` | `build_knowledge_graph.py` |
| `models/kg_tuning_results.json` | `build_knowledge_graph.py` |
| `models/hybrid_best_params.json` | `build_hybrid.py` |
| `models/hybrid_matrix.npz` | `build_hybrid.py` |
| `models/hybrid_grid_results.json` | `build_hybrid.py` |
| `results/metrics.json` | `evaluate.py` |
| `results/summary.md` | `evaluate.py` |

## 🔜 Bước tiếp theo

Chạy các script theo thứ tự để sinh output models và kết quả đánh giá:

```bash
python src/features/build_tfidf.py
python src/features/build_spmi.py
python src/features/build_knowledge_graph.py
python src/features/build_hybrid.py
python src/evaluation/evaluate.py