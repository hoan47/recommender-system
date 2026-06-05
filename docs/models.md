# 🧠 Mô hình gợi ý sản phẩm — Recommender System (Global / Item-Oriented)

## Hướng tiếp cận

**Global (Item-Oriiented):** Xây dựng ma trận quan hệ giữa các sản phẩm dựa trên toàn bộ dữ liệu lịch sử (prior), không phân biệt theo từng user.

## Danh sách models

1. **Content-Based (CB)**
2. **Collaborative Filtering (SPMI)**
3. **Knowledge Graph (KG)**
4. **Hybrid**

---

## 1. Content-Based (CB)

**Mục đích:** Tìm sản phẩm **GIỐNG nhau (substitute)** dựa trên nội dung mô tả.

**Vai trò trong hệ thống:** Dùng để **LOẠI** sản phẩm tương tự khỏi danh sách gợi ý của các model khác (SPMI, KG). Không phải model gợi ý chính, chỉ là baseline để so sánh và làm bộ lọc.

**Dữ liệu đầu vào:**
| File | Cột dùng | Mục đích |
|------|----------|----------|
| `products.csv` | product_id, product_name | Text chính cho TF-IDF |
| `departments.csv` | department_id, department | Gắn tên ngành hàng vào document |

> **Không dùng:** `aisles.csv` (kệ hàng là vị trí vật lý, không phải phân loại toàn cục)

**Feature engineering:**
```
Với mỗi sản phẩm:
  document = product_name + " " + department_name
  
Ví dụ:
  product_id=1 → "Chocolate Sandwich Cookies snacks cookies"
```

**Thuật toán:**
1. TF-IDF vectorize documents (ngram_range=(1,2), max_features=10,000)
2. Cosine similarity giữa TF-IDF vectors
3. Output: ma trận similarity (49,688 × 49,688)

**Output:** `models/tfidf_matrix.npz`, `models/item_similarity_cb.npz`, `models/tfidf_vectorizer.pkl`

**Dùng prior/train/test?** Không dùng. CB chỉ dùng products.csv + departments.csv, không cần interaction data.

---

## 2. Collaborative Filtering (SPMI)

**Mục đích:** Tìm sản phẩm **MUA KÈM (complementary)** dựa trên co-occurrence toàn cục.

**Vai trò trong hệ thống:** Model gợi ý chính cho complementary products, tìm sản phẩm hay được mua cùng nhau.

**Dữ liệu đầu vào:**
| File | Cột dùng | Mục đích |
|------|----------|----------|
| `order_products__prior.csv` | order_id, product_id | Đếm co-occurrence (32.4M records) |
| `order_products__train.csv` | order_id, product_id | Tune threshold + đánh giá |
| `products.csv` | product_id | Map tên sản phẩm |

**Thuật toán:**
1. Đếm co-occurrence từ prior (chunk-based, tránh tràn bộ nhớ)
2. Tính PMI: `PMI(A,B) = log(P(A,B) / (P(A) * P(B)))`
3. Tính SPMI: `SPMI(A,B) = max(PMI(A,B) - log(k), 0)` với k là threshold
4. Chỉ giữ spmi > 0 → sparse matrix

**Tune threshold k trên train:**
```
for k in [1, 2, 3, 5, 10]:
    spmi = build_spmi(cooc_matrix, k)
    hit_rate = evaluate(spmi, train)
Chọn k tốt nhất
```

**Đánh giá cuối trên test:** Recall@K, NDCG@K, MAP@K

**Output:** `models/cooc_matrix.npz`, `models/spmi_matrix.npz`, `models/spmi_best_k.json`

---

## 3. Knowledge Graph (KG)

**Mục đích:** Tìm sản phẩm **LIÊN QUAN qua đồ thị** dựa trên cấu trúc product-department và co-occurrence.

**Vai trò trong hệ thống:** Model gợi ý phụ cho complementary products, bổ sung khi SPMI không đủ mạnh (ít dữ liệu).

**Dữ liệu đầu vào:**
| File | Cột dùng | Mục đích |
|------|----------|----------|
| `order_products__prior.csv` | order_id, product_id | Edge co_purchase giữa các products |
| `products.csv` | product_id, department_id | Kết nối product → department |
| `departments.csv` | department_id, department | Node department |

**Cấu trúc đồ thị:**
- **Nodes:** product (49,688) + department (21)
- **Edges:**
  - `(product_A) — [co_purchase] → (product_B)` — nếu SPMI > 0
  - `(product) — [belongs_to] → (department)` — từ products.csv

**Hướng xử lý (Global):** Dùng node2vec để học product embeddings (128-d), sau đó tính similarity bằng cosine giữa các embeddings.

**Output:** `models/kg_embeddings.npy`, `models/kg_best_params.json`

---

## 4. Hybrid

**Mục đích:** Kết hợp SPMI + KG để tận dụng ưu điểm của cả 2, dùng CB làm bộ lọc loại substitute.

**Công thức:**
```
final_score(A → B) = α * spmi_score(A,B) + β * kg_sim(A,B)

Nếu cb_sim(A,B) > threshold → final_score = 0 (loại substitute)
```

**Tune trọng số α, β, và cb_threshold trên train:**
```
for α, β in grid_search:
    score = α * spmi + β * kg
    score = filter_cb(score, cb_sim, threshold)
    hit = evaluate(score, train)
Chọn (α, β, threshold) tốt nhất
```

**Đánh giá cuối trên test:** Recall@K, NDCG@K, MAP@K

**Output:** `models/hybrid_weights.json`

---

## Pipeline tổng thể

```
DATA FLOW:
  products.csv + departments.csv  ──────────► CB (TF-IDF) ──────────► item_similarity_cb
      │
  order_products__prior.csv ──┬──► SPMI (co-occurrence → PMI) ──► spmi_matrix
                              │
                              └──► KG (graph → node2vec) ──────► kg_embeddings
      │
  order_products__train.csv ──► TUNE params (shift k, α, β, threshold)
      │
  order_products__test.csv  ──► EVALUATE final metrics

HYBRID:
  final_score = α * spmi + β * kg   (filtered by cb_sim)
```

## Phân chia dữ liệu

| Tập | Số đơn | Vai trò |
|-----|--------|---------|
| **prior** (94%) | 3,214,874 | Xây dựng toàn bộ model: co-occurrence matrix, graph, TF-IDF |
| **train** (3.8%) | 131,209 | Tune hyperparameters: shift SPMI, params KG, weights Hybrid |
| **test** (2.2%) | 75,000 | Đánh giá cuối: Recall@K, NDCG@K, MAP@K |