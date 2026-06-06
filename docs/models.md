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

**Vai trò trong hệ thống:** Dùng để **LOẠI** sản phẩm tương tự khỏi danh sách gợi ý của các model khác (SPMI, KG). Không phải model gợi ý chính, chỉ là baseline để so sánh và làm bộ lọc. Ngoài ra, CB còn là fallback cho sản phẩm long-tail (freq < 5) — khi SPMI không đủ tin cậy.

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
1. Tokenize + build vocabulary (unigram + bigram, top max_features=10,000 theo DF)
2. TF (sublinear: 1 + log(tf)) × IDF (smooth) → L2 normalize
3. Cosine similarity: chunked dot product + norm → sparse top-K
4. Output: ma trận similarity (49,688 × 49,688)

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
| `order_products__train.csv` | order_id, product_id | Ground truth cho **CẢ train và test** (tách qua `orders.csv[eval_set]`) |
| `orders.csv` | order_id, eval_set | Phân biệt train/test (eval_set='train'/'test') |
| `products.csv` | product_id | Map tên sản phẩm |

**Thuật toán:**
1. Đếm co-occurrence từ prior (chunk-based, tránh tràn bộ nhớ)
2. Tính PMI: `PMI(A,B) = log(P(A,B) / (P(A) * P(B)))` 
- Trong đó:
    P(A, B) là xác suất cả hai sản phẩm $A$ và $B$ cùng xuất hiện trong một đơn hàng
    P(A) Xác suất tìm thấy sản phẩm A trong một đơn hàng ngẫu nhiên, P(B) tương tự
    PMI = 0: A và B độc lập. Không liên quan tới nhau.
    PMI > 0: A và B có mối quan hệ tương hỗ mạnh mẽ (sản phẩm mua kèm thực sự).
    PMI < 0: A và B "nói không" với nhau. Người mua A thường sẽ chủ động không mua B.

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
| `order_products__prior.csv` | order_id, product_id | Edge co_purchase giữa các products (qua SPMI) |
| `products.csv` | product_id, department_id | Kết nối product → department |
| `departments.csv` | department_id, department | Node department |
| `models/spmi_matrix.npz` | (từ bước SPMI) | Lọc edges: chỉ giữ cặp có SPMI > 0 |

**Cấu trúc đồ thị:**
- **Nodes:** product (49,688) + department (21)
- **Edges:**
  - `(product_A) — [co_purchase] → (product_B)` — **chỉ giữ cặp có SPMI > 0**, weight = SPMI value. Dùng SPMI thay vì co-occurrence count giúp lọc nhiễu, giảm số edges.
  - `(product) — [belongs_to] → (department)` — weight = 1.0, từ products.csv

**Hướng xử lý (Global):** Node2vec (random walks + skip-gram với negative sampling) để học product embeddings, sau đó tính similarity bằng cosine giữa các embeddings. Dùng numpy + networkx. Tune params (walk_length, dimensions, num_walks) trên train.

**Grid search space:**
- `walk_length`: [10, 20, 30]
- `dimensions`: [64, 128]
- `num_walks`: [100, 200]

**Output:** `models/kg_embeddings.npy`, `models/kg_best_params.json`, `models/kg_similarity.npz` (sparse cosine similarity)

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
  order_products__train.csv + orders.csv ──► TUNE params (train) + EVALUATE (test)

HYBRID:
  final_score = α * spmi + β * kg   (filtered by cb_sim)
```

## Phân chia dữ liệu

| Tập | Số đơn | Vai trò |
|-----|--------|---------|
| **prior** (94%) | 3,214,874 | Xây dựng toàn bộ model: co-occurrence matrix, graph, TF-IDF |
| **train** (3.8%) | 131,209 | Tune hyperparameters: shift SPMI, params KG, weights Hybrid |
| **test** (2.2%) | 75,000 | Đánh giá cuối: Recall@K, NDCG@K, MAP@K |