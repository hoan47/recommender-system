# 📋 Kế hoạch Implement — Global Models (Item-Oriented)

## Thông tin chung

| Mục | Giá trị |
|-----|---------|
| Dự án | Recommender System — Instacart Market Basket Analysis |
| Hướng tiếp cận | Global (Item-Oriented) |
| Models | CB, SPMI, KG, Hybrid |
| Dữ liệu chính | `order_products__prior.csv` (32.4M records) |
| Ngôn ngữ | Python 3.9+ |
| Thư mục output | `models/`, `results/` |

---

## 🗂️ Cấu trúc thư mục code cần tạo

```
src/
├── utils/
│   └── data_loader.py         ← Load dữ liệu từ data/
├── features/
│   ├── build_tfidf.py         ← Content-Based (CB)
│   ├── build_spmi.py          ← Collaborative Filtering (SPMI)
│   ├── build_knowledge_graph.py  ← Knowledge Graph (KG)
│   └── build_hybrid.py        ← Hybrid
└── evaluation/
    └── evaluate.py            ← Đánh giá tất cả models
```

---

## ⏱️ Thứ tự thực hiện & Phụ thuộc

```
data_loader.py ──────┬──► build_tfidf.py (CB)
                     │
                     ├──► build_spmi.py (SPMI)
                     │
                     └──► build_knowledge_graph.py (KG)
                              │
build_tfidf.py  ──────────────┤
build_spmi.py   ──────────────┼──► build_hybrid.py (Hybrid)
build_kg.py     ──────────────┤
                              │
                     ┌────────┘
                     ▼
               evaluate.py ─────► So sánh tất cả models
```

| Thứ tự | File | Phụ thuộc | Giải thích |
|--------|------|-----------|------------|
| 1 | `data_loader.py` | Không | Load + xử lý dữ liệu |
| 2 | `build_tfidf.py` (CB) | data_loader | CB độc lập, chỉ cần products.csv + departments.csv |
| 3 | `build_spmi.py` (SPMI) | data_loader | SPMI độc lập, chỉ cần prior interactions |
| 4 | `build_knowledge_graph.py` (KG) | data_loader | KG độc lập, xây graph từ prior + products |
| 5 | `build_hybrid.py` (Hybrid) | CB + SPMI + KG | Kết hợp 3 model trên |
| 6 | `evaluate.py` | CB + SPMI + KG + Hybrid | Đánh giá tất cả model trên test |

> **Giải thích phụ thuộc:**
> - **CB, SPMI, KG độc lập với nhau** — mỗi model xây dựng từ prior + metadata riêng
> - KG CÓ THỂ dùng SPMI để lọc edges (chỉ giữ co_purchase nếu SPMI > 0), nhưng KHÔNG bắt buộc — có thể dùng co-occurrence counts trực tiếp
> - Hybrid là model kết hợp, cần output của cả 3 model kia
> - Evaluate cần tất cả model matrices để so sánh

---

## 📊 Cách phân biệt Train và Test trong cùng 1 file

### Vấn đề
Dataset không có file `order_products__test.csv`. Chỉ có 2 file:
- `order_products__prior.csv` — interactions của prior orders (94%)
- `order_products__train.csv` — interactions của **CẢ train (3.8%) VÀ test (2.2%)**

### Cách phân biệt
Dùng `orders.csv` có cột `eval_set`:
```python
# Load orders
orders = pd.read_csv('data/orders.csv')

# Lọc order_id theo eval_set
train_order_ids = set(orders[orders['eval_set'] == 'train']['order_id'])  # 131,209
test_order_ids  = set(orders[orders['eval_set'] == 'test']['order_id'])   # 75,000

# Load ALL order products (prior + train file)
prior_products = pd.read_csv('data/order_products__prior.csv')
train_products = pd.read_csv('data/order_products__train.csv')

# Tách ground truth
train_gt = train_products[train_products['order_id'].isin(train_order_ids)]
test_gt  = train_products[train_products['order_id'].isin(test_order_ids)]
```

---

## 📝 Chi tiết từng bước

### Bước 1: `src/utils/data_loader.py`

**Mục đích:** Load và xử lý tất cả file dữ liệu, cung cấp interface thống nhất cho các model.

**Functions cần viết:**

```python
def load_products():
    """Đọc products.csv + departments.csv. Bỏ qua aisles.csv.
    Output: DataFrame[product_id, product_name, department_id, department]"""

def load_orders(eval_set=None):
    """Đọc orders.csv. Lọc theo eval_set nếu cần.
    Output: DataFrame[order_id, user_id, eval_set, order_number, ...]"""

def load_order_products(file_type='prior'):
    """Đọc order_products__prior.csv hoặc train.csv
    Output: DataFrame[order_id, product_id, add_to_cart_order, reordered]"""

def load_data_for_model(model_name):
    """
    Trả về dữ liệu phù hợp cho từng model:
    - 'cb': products_df
    - 'spmi': (prior_products_df, train_gt_df, test_gt_df)
    - 'kg': (prior_products_df, products_df)
    - 'hybrid': (cb_sim, spmi, kg_sim, train_gt_df, test_gt_df)
    """
```

**Lưu ý kỹ thuật:**
- Đọc với `encoding='utf-8'`
- Dùng `csv.DictReader` (có dấu phẩy trong ngoặc kép)
- `order_products__prior.csv` 32.4M records → đọc chunk khi cần

**File output:** `src/utils/__init__.py` (để trống)

---

### Bước 2: `src/features/build_tfidf.py` (Content-Based)

**Độc lập** — không phụ thuộc model nào khác. Chỉ cần `data_loader`.

**Steps:**

1. Load products + departments từ data_loader
2. Tạo document cho mỗi sản phẩm: `product_name + " " + department`
3. TF-IDF vectorize: `ngram_range=(1,2)`, `max_features=10000`, `stop_words='english'`
4. Cosine similarity → sparse matrix (49,688 × 49,688)
5. Save:
   - `models/tfidf_matrix.npz`
   - `models/item_similarity_cb.npz`
   - `models/tfidf_vectorizer.pkl`

**Dùng prior/train/test?** Không dùng — CB thuần text.

---

### Bước 3: `src/features/build_spmi.py` (Collaborative Filtering)

**Độc lập** — không phụ thuộc model nào khác. Chỉ cần `data_loader`.

**Steps:**

1. Load prior interactions từ data_loader
2. Đếm co-occurrence từ prior (chunk-based, 32.4M records)
   - Với mỗi order, lấy list products → đếm tất cả cặp (A,B)
   - Dùng `scipy.sparse.dok_matrix` để incremental building
3. Tính PMI: `PMI(A,B) = log(cooc[A][B] * total_orders / (freq[A] * freq[B]))`
4. Tính SPMI: `SPMI(A,B) = max(PMI(A,B) - log(k), 0)`
5. Tune k trên **train** (vòng lặp k=1,2,3,5,10) → chọn k tốt nhất
6. Evaluate cuối trên **test** (Recall@K, NDCG@K, MAP@K)
7. Save:
   - `models/cooc_matrix.npz`
   - `models/spmi_matrix.npz` (sparse, chỉ giữ spmi > 0)
   - `models/spmi_best_k.json`

**Dùng prior/train/test?**
- prior (100%) → xây co-occurrence
- train → tune threshold k
- test → eval cuối

---

### Bước 4: `src/features/build_knowledge_graph.py` (KG)

**Độc lập** — có thể chạy không cần SPMI (dùng co-occurrence counts thay vì SPMI để tạo edges).

**Steps:**

1. Load prior + products + departments từ data_loader
2. Xây dựng đồ thị:
   - Nodes: product (49,688) + department (21)
   - Edges:
     - `(product_A) — [co_purchase] → (product_B)` — weight = co-occurrence count
     - `(product) — [belongs_to] → (department)` — weight = 1.0
3. Học node2vec embeddings (128-d):
   ```python
   from node2vec import Node2Vec
   node2vec = Node2Vec(graph, dimensions=128, walk_length=20, num_walks=200)
   model = node2vec.fit(window=10, min_count=1)
   ```
4. Cosine similarity giữa product embeddings
5. Tune params (walk_length, dimensions) trên **train**
6. Evaluate cuối trên **test**
7. Save:
   - `models/kg_embeddings.npy`
   - `models/kg_best_params.json`
   - `models/kg_similarity.npz` (sparse, chỉ product-product)

**Dùng prior/train/test?**
- prior (100%) + products → xây graph + học node2vec
- train → tune params
- test → eval cuối

---

### Bước 5: `src/features/build_hybrid.py` (Hybrid)

**Phụ thuộc:** CB + SPMI + KG đã hoàn thành.

**Steps:**

1. Load CB, SPMI, KG từ files
2. Công thức:
   ```
   final_score = α * spmi + β * kg
   Nếu cb_sim > threshold → final_score = 0 (loại substitute)
   ```
3. Grid search α (0.0–1.0, step 0.2), β (0.0–1.0, step 0.2), cb_threshold (0.7, 0.8, 0.9) trên **train**
4. Evaluate cuối trên **test** với best params
5. Save: `models/hybrid_weights.json`

**Dùng prior/train/test?**
- prior → CB/SPMI/KG đã xây sẵn
- train → tune α, β, threshold
- test → eval cuối

---

### Bước 6: `src/evaluation/evaluate.py`

**Phụ thuộc:** Cần ít nhất 1 model matrix để evaluate.

**Functions:**

```python
def recall_at_k(recommended, ground_truth, k):
    """Tỉ lệ sản phẩm đúng trong top-K"""
    
def ndcg_at_k(recommended, ground_truth, k):
    """Normalized Discounted Cumulative Gain @ K"""
    
def map_at_k(recommended, ground_truth, k):
    """Mean Average Precision @ K"""
    
def evaluate_model(model_matrix, test_data, ks=[5, 10, 20]):
    """
    Cơ chế leave-one-out:
    - Với mỗi order trong test:
        - Lấy tất cả products trong order
        - Với mỗi product A:
            - Lấy top-K products từ model_matrix[A]
            - Ground truth = các products còn lại trong order
    - Tính trung bình Recall@K, NDCG@K, MAP@K
    """

def compare_models(models_dict, test_data):
    """
    So sánh CB vs SPMI vs KG vs Hybrid
    Output: DataFrame → results/metrics.json
    """
```

---

## 🧪 Evaluation Protocol Chi Tiết

### Cơ chế đánh giá (Leave-One-Out per Product)

```
Với mỗi order O trong test set:
  products = [P1, P2, P3, ..., Pn]  # từ order_products__train.csv, filter by test order_ids
  
  Với mỗi Pi làm query:
    - ground_truth = {P1, ..., Pi-1, Pi+1, ..., Pn}  # các sản phẩm CÒN LẠI trong order
    - recommendations = model.top_k(Pi)  # top-K sản phẩm dựa trên model matrix
  
  - Tính recall@K cho mỗi query
  → Trung bình trên tất cả queries trong tất cả orders
```

### Metrics báo cáo

| K | Recall | NDCG | MAP |
|---|--------|------|-----|
| 5 | ✅ | ✅ | ✅ |
| 10 | ✅ | ✅ | ✅ |
| 20 | ✅ | ✅ | ✅ |

---

## ⚠️ Lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|--------|-----------|
| `order_products__prior.csv` 32.4M records | Đọc theo chunk (chunksize=500K) |
| Ma trận 50K×50K dense ~20GB | Luôn dùng `scipy.sparse` |
| encoding file CSV | `encoding='utf-8'` |
| Dấu phẩy trong ngoặc kép | Dùng `csv.DictReader` |
| Ground truth train/test chung 1 file | Tách bằng `orders.csv[eval_set]` |
| node2vec training lâu | Giảm walk_length, num_walks khi test |
| RAM không đủ | Dùng `dok_matrix` hoặc `lil_matrix` cho incremental |

---

## ✅ Checklist hoàn thành

- [ ] `src/utils/data_loader.py` — load được tất cả dữ liệu, tách train/test từ orders.csv
- [ ] `src/features/build_tfidf.py` — TF-IDF + cosine similarity
- [ ] `src/features/build_spmi.py` — co-occurrence → PMI → SPMI → tune → eval
- [ ] `src/features/build_knowledge_graph.py` — graph → node2vec → similarity
- [ ] `src/features/build_hybrid.py` — hybrid score → grid search → eval
- [ ] `src/evaluation/evaluate.py` — Recall@K, NDCG@K, MAP@K cho tất cả models
- [ ] `results/metrics.json` — so sánh tất cả models