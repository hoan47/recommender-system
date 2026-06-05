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
├── __init__.py                     ← File trống, đánh dấu package
├── utils/
│   ├── __init__.py                 ← File trống
│   └── data_loader.py             ← Load dữ liệu từ data/
├── features/
│   ├── __init__.py                 ← File trống
│   ├── build_tfidf.py             ← Content-Based (CB)
│   ├── build_spmi.py              ← Collaborative Filtering (SPMI)
│   ├── build_knowledge_graph.py   ← Knowledge Graph (KG)
│   └── build_hybrid.py            ← Hybrid
└── evaluation/
    ├── __init__.py                 ← File trống
    └── evaluate.py                ← Đánh giá tất cả models
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
| 4 | `build_knowledge_graph.py` (KG) | data_loader + SPMI (để tạo edges) | KG cần SPMI > 0 để tạo edges co_purchase, nhưng tránh circular dependency: SPMI build xong trước, rồi KG load spmi_matrix để lọc edges |
| 5 | `build_hybrid.py` (Hybrid) | CB + SPMI + KG | Kết hợp 3 model trên |
| 6 | `evaluate.py` | CB + SPMI + KG + Hybrid | Đánh giá tất cả model trên test (**chỉ dùng test 1 lần duy nhất**) |

> **Giải thích phụ thuộc:**
> - **CB, SPMI độc lập với nhau** — mỗi model xây dựng từ prior + metadata riêng
> - **KG phụ thuộc SPMI** — dùng `spmi_matrix.npz` để tạo edges: chỉ giữ cặp (A,B) có SPMI > 0 làm edge co_purchase trong đồ thị. Điều này giúp lọc nhiễu và giảm số edges đáng kể so với dùng co-occurrence counts thô.
> - **Hybrid là model kết hợp**, cần output của cả 3 model kia
> - **Evaluate chỉ dùng test 1 lần duy nhất** — các bước trước chỉ tune trên train, không đụng đến test

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

def load_train_test_split():
    """
    Tách ground truth từ order_products__train.csv dựa trên orders.csv[eval_set].
    Output: (train_gt_df, test_gt_df)
    """

def load_data_for_model(model_name):
    """
    Trả về dữ liệu phù hợp cho từng model:
    - 'cb': products_df
    - 'spmi': (prior_products_df, train_gt_df, test_gt_df)
    - 'kg': (prior_products_df, products_df)
    """
    # Lưu ý: KHÔNG đưa 'hybrid' vào đây vì hybrid cần load model outputs
    # từ các file đã được tạo bởi CB, SPMI, KG
```

**Lưu ý kỹ thuật:**
- Đọc với `encoding='utf-8'`
- Dùng `csv.DictReader` (có dấu phẩy trong ngoặc kép)
- `order_products__prior.csv` 32.4M records → đọc chunk khi cần (chunksize=500K)
- Tạo thư mục `src/utils/`, `src/features/`, `src/evaluation/` trước khi viết file

**File output cần tạo:**
- `src/__init__.py` (trống)
- `src/utils/__init__.py` (trống)
- `src/features/__init__.py` (trống)
- `src/evaluation/__init__.py` (trống)

---

### Bước 2: `src/features/build_tfidf.py` (Content-Based)

**Độc lập** — không phụ thuộc model nào khác. Chỉ cần `data_loader`.

**Steps:**

1. Load products + departments từ data_loader
2. Tạo document cho mỗi sản phẩm: `product_name + " " + department`
3. TF-IDF vectorize: `ngram_range=(1,2)`, `max_features=10000`, `stop_words='english'`
4. Cosine similarity → sparse matrix (49,688 × 49,688). Chỉ giữ top-N hoặc similarity > ngưỡng để giảm kích thước.
5. Save:
   - `models/tfidf_matrix.npz`
   - `models/item_similarity_cb.npz`
   - `models/tfidf_vectorizer.pkl`

**Dùng prior/train/test?** Không dùng — CB thuần text.

**Edge cases:**
- Sản phẩm có product_name rỗng hoặc NaN → dùng "unknown product " + department
- Department rỗng → dùng "unknown department"

---

### Bước 3: `src/features/build_spmi.py` (Collaborative Filtering)

**Độc lập** — không phụ thuộc model nào khác. Chỉ cần `data_loader`.

**Steps:**

1. Load prior interactions từ data_loader
2. Đếm co-occurrence từ prior (chunk-based, 32.4M records)
   - Với mỗi order, lấy list products → đếm tất cả cặp (A,B) trong cùng order
   - Dùng `scipy.sparse.dok_matrix` hoặc `lil_matrix` để incremental building
3. Tính PMI:
   ```
   PMI(A,B) = log(cooc[A][B] * total_prior_orders / (freq[A] * freq[B]))
   
   Trong đó:
   - total_prior_orders = 3,214,874 (số lượng orders trong prior, KHÔNG phải tổng interactions)
   - freq[A] = số orders chứa sản phẩm A trong prior (document frequency, không phải tổng lượt mua)
   ```
4. Tính SPMI: `SPMI(A,B) = max(PMI(A,B) - log(k), 0)`
5. Tune k trên **train** (vòng lặp k=1, 2, 3, 5, 10) → chọn k tốt nhất
   - **Chỉ dùng train để tune** — không đụng đến test
   - Dùng in-sample evaluation: với mỗi order trong train, leave-one-out để tính hit-rate
6. Save:
   - `models/cooc_matrix.npz`
   - `models/spmi_matrix.npz` (sparse, chỉ giữ spmi > 0)
   - `models/spmi_best_k.json`

**Dùng prior/train/test?**
- prior (100%) → xây co-occurrence
- train → tune threshold k (không dùng test)
- test → chỉ dùng ở `evaluate.py` (bước 6)

**Edge cases:**
- Sản phẩm chưa từng xuất hiện trong prior (11/49,688) → cooc[A][*] = 0, SPMI = 0 với mọi cặp
- Sản phẩm long-tail (freq < 5) → PMI không đáng tin cậy, SPMI ≈ 0 sau shift

---

### Bước 4: `src/features/build_knowledge_graph.py` (KG)

**Phụ thuộc SPMI** — cần SPMI đã hoàn thành (bước 3) để tạo edges co_purchase. Có thể chạy ngay sau SPMI, song song với CB.

**Steps:**

1. Load prior + products + departments từ data_loader
2. Load `models/spmi_matrix.npz` từ bước 3 để tạo edges
3. Xây dựng đồ thị:
   - Nodes: product (49,688) + department (21)
   - Edges:
     - `(product_A) — [co_purchase] → (product_B)` — **chỉ giữ cặp có SPMI > 0**, weight = SPMI value. Dùng SPMI thay vì co-occurrence count giúp lọc nhiễu, giảm số edges đáng kể.
     - `(product) — [belongs_to] → (department)` — weight = 1.0
4. Học node2vec embeddings (tự code, KHÔNG dùng thư viện node2vec/gensim):
   - **Random Walks (node2vec strategy):**
     - Với mỗi node, thực hiện `num_walks` lần random walk độ dài `walk_length`
     - Transition probability: p=1 (return), q=1 (in-out), α_pq(t, x) × w_vx
   - **Skip-Gram với Negative Sampling:**
     - Input: các walks (sequences of node IDs)
     - Context window = 10, negative samples = 5
     - Loss = log σ(pos_dot) + Σ log σ(-neg_dot)
     - Optimize bằng SGD, learning rate giảm dần
   - **Chỉ dùng numpy + networkx**, không import `node2vec`, `gensim`, hay bất kỳ thư viện ML nào
5. Cosine similarity giữa product embeddings → sparse matrix (chỉ product-product)
6. Tune params trên **train** (in-sample, không dùng test):
   - **Grid search space:**
     - `walk_length`: [10, 20, 30]
     - `dimensions`: [64, 128]
     - `num_walks`: [100, 200]
   - Chọn params tốt nhất dựa trên hit-rate@K trên train
7. Save:
   - `models/kg_embeddings.npy` — (49,688 × best_dimensions)
   - `models/kg_best_params.json` — best walk_length, dimensions, num_walks
   - `models/kg_similarity.npz` — sparse cosine similarity (product × product)

**Dùng prior/train/test?**
- prior (100%) + products → xây graph + học node2vec (không dùng test)
- train → tune params (không dùng test)
- test → chỉ dùng ở `evaluate.py` (bước 6)

**Edge cases:**
- Sản phẩm không có bất kỳ edge co_purchase nào (spmi=0 với mọi sản phẩm) → chỉ có edge belongs_to đến department, embedding phụ thuộc vào department node
- Graph quá lớn → giảm num_walks xuống 50-100 khi test code

---

### Bước 5: `src/features/build_hybrid.py` (Hybrid)

**Phụ thuộc:** CB + SPMI + KG đã hoàn thành.

**Steps:**

1. Load CB, SPMI, KG matrices từ files:
   ```python
   spmi = load_sparse('models/spmi_matrix.npz')
   kg_sim = load_sparse('models/kg_similarity.npz')
   cb_sim = load_sparse('models/item_similarity_cb.npz')
   ```

2. Normalize scores trước khi combine (quan trọng vì scale khác nhau):
   ```python
   # SPMI có thể > 1, kg_sim (cosine) trong [0, 1], cb_sim (cosine) trong [0, 1]
   # Normalize SPMI về [0, 1] dùng min-max hoặc z-score
   spmi_norm = spmi / spmi.max()  # đơn giản nhất
   ```

3. Công thức:
   ```
   final_score(A → B) = α * spmi_norm(A,B) + β * kg_sim(A,B)
   Nếu cb_sim(A,B) > cb_threshold → final_score = 0 (loại substitute)
   ```

4. Grid search α (0.0–1.0, step 0.2), β (0.0–1.0, step 0.2), cb_threshold (0.7, 0.8, 0.9) trên **train** (in-sample)
5. Save best params: `models/hybrid_weights.json` — {α, β, cb_threshold, best_score}
6. **Không eval trên test** — để dành cho `evaluate.py`

**Dùng prior/train/test?**
- prior → CB/SPMI/KG đã xây sẵn
- train → tune α, β, threshold (không dùng test)
- test → chỉ dùng ở `evaluate.py` (bước 6)

**Edge cases:**
- Sản phẩm A có spmi=0 và kg_sim=0 với mọi B → final_score = 0 với mọi B, recommend rỗng
- cb_sim cao (> threshold) → loại hết tất cả candidates → fallback: chỉ dùng spmi + kg, bỏ qua filter

---

### Bước 6: `src/evaluation/evaluate.py`

**Phụ thuộc:** Cần ít nhất 1 model matrix để evaluate. **Đây là nơi DUY NHẤT được dùng test set.**

**Functions:**

```python
def recall_at_k(recommended, ground_truth, k):
    """Tỉ lệ sản phẩm đúng trong top-K"""

def ndcg_at_k(recommended, ground_truth, k):
    """Normalized Discounted Cumulative Gain @ K"""

def map_at_k(recommended, ground_truth, k):
    """Mean Average Precision @ K"""

def evaluate_model(model_matrix, test_data, ks=[5, 10, 20], model_name=""):
    """
    Cơ chế leave-one-out:
    - Với mỗi order trong test:
        - Lấy tất cả products trong order
        - Với mỗi product A:
            - Lấy top-K products từ model_matrix[A]
            - Ground truth = các products còn lại trong order
            - Bỏ qua order có < 2 sản phẩm (không có ground truth)
    - Tính trung bình Recall@K, NDCG@K, MAP@K
    """

def compare_models(models_dict, test_data):
    """
    So sánh CB vs SPMI vs KG vs Hybrid
    Input: models_dict = {
        'CB': cb_matrix,
        'SPMI': spmi_matrix,
        'KG': kg_matrix,
        'Hybrid': hybrid_matrix
    }
    Output: DataFrame → results/metrics.json
    """
```

---

## 🧪 Evaluation Protocol Chi Tiết

### Cơ chế đánh giá (Leave-One-Out per Product)

```
Với mỗi order O trong test set:
  products = [P1, P2, P3, ..., Pn]  # từ order_products__train.csv, filter by test order_ids
  
  Nếu n < 2 → bỏ qua order này (không có ground truth)
  
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

### Quy tắc dùng tập dữ liệu

| Tập | Mục đích | Ai dùng | Khi nào |
|-----|----------|---------|---------|
| **prior** (3,214,874 đơn) | Xây dựng toàn bộ model | CB, SPMI, KG | Bước 2, 3, 4 |
| **train** (131,209 đơn) | Tune hyperparameters (in-sample) | SPMI, KG, Hybrid | Bước 3, 4, 5 |
| **test** (75,000 đơn) | **Đánh giá cuối — CHỈ 1 LẦN** | `evaluate.py` | Bước 6 |

> ⚠️ **Không dùng test để tune params.** Test chỉ để báo cáo metrics cuối cùng. Dùng test nhiều lần sẽ gây data leakage và kết quả không còn ý nghĩa thống kê.

---

## ⚠️ Lưu ý kỹ thuật

| Vấn đề | Giải pháp |
|--------|-----------|
| `order_products__prior.csv` 32.4M records | Đọc theo chunk (chunksize=500K) |
| Ma trận 50K×50K dense ~20GB | Luôn dùng `scipy.sparse` (`csr_matrix`, `dok_matrix`, `lil_matrix`) |
| encoding file CSV | `encoding='utf-8'` |
| Dấu phẩy trong ngoặc kép | Dùng `csv.DictReader` thay vì split thủ công |
| Ground truth train/test chung 1 file | Tách bằng `orders.csv[eval_set]` |
| node2vec training lâu | Giảm walk_length, num_walks khi test code |
| RAM không đủ | Dùng `dok_matrix` hoặc `lil_matrix` cho incremental building |
| SPMI và cosine khác scale | Normalize về [0,1] trước khi combine trong Hybrid |
| 11 sản phẩm chưa từng xuất hiện trong prior | Co-occurrence = 0 với mọi cặp → SPMI = 0, KG edges rỗng |
| Order test có < 2 sản phẩm | Bỏ qua khi evaluate (không có ground truth) |
| Sản phẩm long-tail (freq < 5) | SPMI ≈ 0 → dùng CB làm fallback cho similarity |

---

## ✅ Checklist hoàn thành (Sau mỗi task phải cập nhật lại progress.md dự án hiện tại đã có gì)

- [ ] Tạo cấu trúc thư mục `src/` + `__init__.py` (utils, features, evaluation)
- [ ] Tạo thư mục `models/`, `results/` (nếu chưa có)
- [ ] `src/utils/data_loader.py` — load được tất cả dữ liệu, tách train/test từ orders.csv
- [ ] `src/features/build_tfidf.py` — TF-IDF + cosine similarity (sparse)
- [ ] `src/features/build_spmi.py` — co-occurrence → PMI → SPMI → tune k trên train → save
- [ ] `src/features/build_knowledge_graph.py` — graph từ SPMI edges → node2vec → tune trên train → save
- [ ] `src/features/build_hybrid.py` — normalize + hybrid score → grid search trên train → save weights
- [ ] `src/evaluation/evaluate.py` — Recall@K, NDCG@K, MAP@K cho tất cả models (dùng test 1 lần)
- [ ] `results/metrics.json` — so sánh tất cả models
