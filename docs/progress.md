# 📊 Tiến độ dự án — Recommender System (Global / Item-Oriented)

Cập nhật lần cuối: 06/05/2026 23:02 (Asia/Saigon)

## ✅ Checklist hoàn thành

- [x] Tạo cấu trúc thư mục `src/` + `__init__.py` (utils, features, evaluation)
- [x] Tạo thư mục `models/`, `results/` (nếu chưa có)
- [x] `src/utils/data_loader.py` — load được tất cả dữ liệu, tách train/test từ orders.csv
- [x] `src/features/build_tfidf.py` — TF-IDF + cosine similarity (numpy+scipy, không sklearn)
- [x] `src/features/build_spmi.py` — co-occurrence → PMI → SPMI → tune k trên train → save
- [x] `src/features/build_knowledge_graph.py` — graph → node2vec (numpy+networkx, không node2vec/gensim/sklearn) → tune trên train → save
- [x] `src/features/build_hybrid.py` — normalize + hybrid score → grid search trên train → save weights
- [x] `src/evaluation/evaluate.py` — Recall@K, NDCG@K, MAP@K cho tất cả models (dùng test 1 lần)
- [x] **Refactor: Loại bỏ thư viện ML có sẵn** — thay thế TF-IDF, node2vec, skip-gram, cosine similarity bằng numpy+scipy (xem [#refactor-ml-libs] bên dưới)
- [ ] `results/metrics.json` — so sánh tất cả models (sẽ có sau khi chạy `evaluate.py`)

## 📁 Hiện trạng file

| File | Trạng thái | Mô tả |
|------|-----------|-------|
| `src/utils/data_loader.py` | ✅ Đã tạo | ~170 dòng, load 5 file CSV, tách train/test |
| `src/features/build_tfidf.py` | ✅ Đã tạo | ~330 dòng, TF-IDF + cosine similarity (numpy+scipy) |
| `src/features/build_spmi.py` | ✅ Đã tạo | ~370 dòng, co-occurrence → SPMI → tune k |
| `src/features/build_knowledge_graph.py` | ✅ Đã tạo | ~520 dòng, graph → random walks → skip-gram → tune params |
| `src/features/build_hybrid.py` | ✅ Đã tạo | ~430 dòng, normalize → grid search α/β/threshold |
| `src/evaluation/evaluate.py` | ✅ Đã tạo | ~370 dòng, Recall@K, NDCG@K, MAP@K |
| `requirements.txt` | ✅ Cập nhật | Đã bỏ scikit-learn, node2vec, gensim |

## 🔄 Refactor: Loại bỏ thư viện ML có sẵn [#refactor-ml-libs]

Thực hiện ngày 06/05/2026 để đảm bảo kiểm soát hoàn toàn thuật toán:

| Thư viện đã bỏ | File bị ảnh hưởng | Giải pháp thay thế |
|----------------|-------------------|-------------------|
| `sklearn.TfidfVectorizer` | `build_tfidf.py` | Tự code: tokenize → vocab (unigram+bigram) → TF (sublinear) × IDF (smooth) → L2 normalize |
| `sklearn.cosine_similarity` | `build_tfidf.py`, `build_knowledge_graph.py` | Tự code: L2 normalize rows → chunked dot product → top-K |
| `node2vec.Node2Vec` | `build_knowledge_graph.py` | Tự code: random walks (node2vec strategy p/q) + skip-gram with negative sampling |
| `gensim` | `build_knowledge_graph.py` | (Không trực tiếp dùng, nhưng là dependency của node2vec) |

**Thư viện được giữ lại:** `numpy`, `scipy.sparse`, `networkx`, `pandas`, `tqdm`

## 📦 Output models (thay đổi so với trước)

| File | Script tạo | Thay đổi |
|------|------------|----------|
| `models/tfidf_matrix.npz` | `build_tfidf.py` | Không đổi |
| `models/item_similarity_cb.npz` | `build_tfidf.py` | Không đổi |
| `models/tfidf_vocab.json` | `build_tfidf.py` | **Thay đổi:** dùng JSON thay vì pickle (.pkl) |
| ~~`models/tfidf_vectorizer.pkl`~~ | — | **Đã bỏ** (không còn vectorizer sklearn) |
| `models/cooc_matrix.npz` | `build_spmi.py` | Không đổi |
| `models/spmi_matrix.npz` | `build_spmi.py` | Không đổi |
| `models/spmi_best_k.json` | `build_spmi.py` | Không đổi |
| `models/kg_embeddings.npy` | `build_knowledge_graph.py` | Không đổi |
| `models/kg_best_params.json` | `build_knowledge_graph.py` | Không đổi |
| `models/kg_similarity.npz` | `build_knowledge_graph.py` | Không đổi |
| `models/kg_tuning_results.json` | `build_knowledge_graph.py` | Không đổi |
| `models/hybrid_best_params.json` | `build_hybrid.py` | Không đổi |
| `models/hybrid_matrix.npz` | `build_hybrid.py` | Không đổi |
| `models/hybrid_grid_results.json` | `build_hybrid.py` | Không đổi |
| `results/metrics.json` | `evaluate.py` | Không đổi |
| `results/summary.md` | `evaluate.py` | Không đổi |

## 🔜 Bước tiếp theo

Chạy các script theo thứ tự để sinh output models và kết quả đánh giá:

```bash
python src/features/build_tfidf.py
python src/features/build_spmi.py
python src/features/build_knowledge_graph.py
python src/features/build_hybrid.py
python src/evaluation/evaluate.py