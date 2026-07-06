# Mục Đích Dự Án: Hệ Thống Gợi Ý Mua Kèm (Bundle Recommendation)

## 1. Giới Thiệu Dự Án

Dự án này nhằm xây dựng một **Hệ thống Gợi ý Mua kèm (Bundle Recommendation System)** dựa trên tập dữ liệu Instacart Market Basket Analysis với hơn **31.9 triệu giao dịch** từ hơn **206,000 người dùng** và **36,181 sản phẩm thực phẩm** (đã loại 13,507 non-food).

Dự án chỉ tập trung vào pipeline model chính tại `scripts/model/` (7 bước: 01→07).

Mục tiêu cuối cùng: Khi khách hàng đang chọn một sản phẩm, hệ thống sẽ đề xuất các sản phẩm khác mà khách hàng có **khả năng cao sẽ mua kèm** trong cùng một giỏ hàng.

---

## 2. Phân Biệt Quan Trọng: Gợi Ý Mua Kèm ≠ Dự Đoán Sản Phẩm Tiếp Theo

### ❌ Dự Đoán Sản Phẩm Tiếp Theo (Next Product Prediction)

| Đặc điểm | Mô tả |
|----------|-------|
| **Đầu vào** | Lịch sử mua hàng của user |
| **Đầu ra** | Sản phẩm tiếp theo user sẽ mua trong tương lai |
| **Tính chất** | **Tuần tự theo thời gian** (sequential/temporal) |
| **Ví dụ** | "Tuần trước mua sữa, tuần này có thể mua bánh mì" |
| **Dữ liệu đánh giá** | Có thể dùng `eval_set=test` (train/test split theo thời gian) |

### ✅ Gợi Ý Mua Kèm (Bundle / Complementary Recommendation)

| Đặc điểm | Mô tả |
|----------|-------|
| **Đầu vào** | Sản phẩm hiện tại đang xem/đã chọn |
| **Đầu ra** | Các sản phẩm thường xuất hiện CÙNG với sản phẩm đó trong cùng giỏ hàng |
| **Tính chất** | **Đồng thời trong giỏ hàng** (co-occurrence trong cùng 1 order) |
| **Ví dụ** | "Mua bia → gợi ý mua kèm snack, thịt nướng" |
| **Dữ liệu đánh giá** | **Phải khảo sát thực tế**, không thể dùng eval_set=test |

### Tại sao eval_set=test KHÔNG phù hợp để đánh giá gợi ý mua kèm?

Dữ liệu `eval_set=test` được tạo ra bằng cách:
1. Lấy **đơn hàng cuối cùng** của mỗi user làm test
2. Giấu một số sản phẩm, yêu cầu model dự đoán các sản phẩm đó

Điều này phù hợp cho bài toán **"dự đoán đơn hàng tiếp theo"**, nhưng **không phù hợp** cho bài toán **"gợi ý mua kèm"** vì:
- Dữ liệu test là **đơn hàng tuần tự**, không phải **mối quan hệ đồng thời giữa các sản phẩm**
- Không có ground truth về "sản phẩm nào nên mua kèm với sản phẩm nào" 
- Mối quan hệ mua kèm là phi tuần tự và cần được đánh giá bởi con người

---

## 3. Chiến Lược Sử Dụng Dữ Liệu

### 3.1 Toàn bộ dữ liệu dùng để TRAIN (đã lọc non-food)

| File | Số dòng (gốc) | Số dòng (sau lọc) | Vai trò |
|------|---------------|-------------------|---------|
| `order_products__prior.csv` | 32,434,490 | ~30.5M | Dữ liệu lịch sử chính → học mối quan hệ sản phẩm |
| `order_products__train.csv` | 1,384,618 | ~1.3M | Bổ sung thêm dữ liệu train |
| **Tổng cộng** | **33,819,108** | **31,919,315** | **Toàn bộ dùng để train model** |
| **Sản phẩm** | 49,688 | **36,181** (food) | **Đã loại 13,507 non-food** |

### 3.2 Không dùng eval_set=test để đánh giá

- `eval_set=test` (~75,000 đơn hàng) sẽ **KHÔNG** được dùng để đánh giá model gợi ý mua kèm
- Lý do: Đây là dữ liệu kiểu sequential, không phải mua kèm

### 3.3 Đánh giá bằng khảo sát thực tế (Human Evaluation)

Thay vào đó, chất lượng gợi ý mua kèm sẽ được đánh giá thông qua **khảo sát người dùng thực tế**. Quy trình khảo sát chi tiết (phương pháp thu thập, cấu trúc dữ liệu, chỉ số đánh giá) được trình bày ở **Mục 5**.

---

## 4. Các Model / Phương Pháp Tiếp Cận

Dự án triển khai các phương pháp gợi ý mua kèm, phân làm **2 nhóm chính**:

### Nhóm 1: Item-Based Collaborative Filtering
Gồm 2 biến thể — cùng học từ co-occurrence trong order (ma trận tương tác user-item):
- **Memory-based (Ochiai + Confidence)** → **Item-CF**: Tính similarity trực tiếp trên binary interaction vector (Cosine similarity)
- **Neural-based (Skip-gram)** → **Item2Vec (Neural CF)**: Học embedding bằng Word2Vec từ context co-occurrence

### Nhóm 2: Graph-Based
- **KGMetapath**: Xây heterogeneous knowledge graph + Metapath Walk → embedding

> 👉 Cả Item-CF và Item2Vec đều là **Item-Based Collaborative Filtering**, chỉ khác biến thể (memory vs neural). KGMetapath thuộc nhóm Graph-Based, không phải CF.

---

### 4.0 Kiến trúc tổng thể

```
                                ┌───────────────────┐
                                │ Sản phẩm đầu vào  │
                                └─────────┬─────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     │                    │                    │
                     ▼                    ▼                    ▼
           ┌──────────────────┐  ┌───────────────────────┐  ┌──────────────┐
            │ Item-CF          │  │ Item2Vec              │  │ KGMetapath   │
            │ (Memory-based    │  │ (Neural Item-Based CF,│  │ (KG          │
            │  Item-Based CF,  │  │  Word2Vec Skip-gram   │  │  embedding)  │
            │  score có hướng) │  │  trên giỏ hàng)       │  │              │
           └────────┬─────────┘  └───────────┬───────────┘  └──────┬───────┘
                   │                     │                   │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │ Co-occurrence    │
                                 │ Ensemble         │
                                 │ weighted score   │
                                 │ ItemCF+I2V+MW    │
                                 └────────┬─────────┘
                                          │
                                          ▼
                      ┌──────────────────────────────┐
                       │ CB Diversity Filter          │
                       │ (loại bỏ substitute:         │
                       │  sản phẩm quá giống          │
                       │  với sản phẩm đầu vào)       │
                       │ Dùng: Ensemble Count + TF-IDF│
                       │ multi-field:                 │
                       │ - product_name (×1.0)        │
                       │ - aisle_vi    (×0.8)         │
                       │ - dept_vi     (×0.6)         │
                       │ Chỉ tính similarity cho      │
                       │ các cặp được đề xuất,        │
                       │ không tính full matrix       │
                      └───────────────┬──────────────┘
                                 │
                                 ▼
                        Top-K gợi ý mua kèm
                        (chỉ complementary,
                         không có substitute)
```

---

### 4.1 Content-Based Filtering (CB) — Diversity Filter

**Vai trò**: Hoạt động như một **bộ lọc hậu xử lý (post-processing filter)** — loại bỏ các sản phẩm **thay thế (substitute)** khỏi kết quả gợi ý của các model co-occurrence. CB **không tham gia ensemble** và **không gợi ý sản phẩm**.

**Vai trò phụ**: Xử lý **cold-start** và **long-tail** — sản phẩm mới hoặc ít xuất hiện vẫn có thể được lọc đúng nhờ thông tin mô tả (tên + aisle + department).

#### 4.1.1 Kiến trúc vector hóa — Ensemble Count + TF-IDF Multi-field

CB sử dụng **2 nhánh vector hóa song song**, kết hợp bằng ensemble:

##### Nhánh 1: TF-IDF Multi-field (có trọng số)

Vector hóa sản phẩm dựa trên **3 trường** với trọng số khác nhau:

| Trường | Trọng số | N-gram range | Max features | Lý do |
|--------|----------|-------------|-------------|-------|
| `product_name` | ×1.0 | (1,2) — unigram + bigram | 15,000 | Quan trọng nhất, tên sản phẩm mô tả chính xác nhất |
| `aisle` | ×0.8 | (1,1) — unigram | 500 | Aisle ngắn (127 loại), chỉ cần unigram |
| `department` | ×0.6 | (1,1) — unigram | 100 | Department rất ngắn (21 loại), chỉ cần unigram |

Công thức:
```python
combined_vector = hstack([
    tfidf(name_texts) * 1.0,
    tfidf(aisle_texts) * 0.8,
    tfidf(dept_texts) * 0.6,
])  # shape: (n_products, ~15,600)
```

Cả 3 trường đều được ghi đè bằng **bản dịch tiếng Việt** (qua `loader.py` → `products_vi.csv`, `aisles_vi.csv`, `departments_vi.csv`) để model hiểu được ngữ nghĩa tiếng Việt.

##### Nhánh 2: Count Vectorizer (Overlap Score)

Chỉ trên `product_name`:

| Tham số | Giá trị |
|---------|---------|
| N-gram range | (1,1) — unigram |
| Max features | 15,000 |
| Metric | Overlap Coefficient |

Overlap Coefficient = `|A ∩ B| / min(|A|, |B|)` — chỉ quan tâm tỷ lệ từ trùng so với bên ít từ hơn, bất chấp độ dài câu.

Ví dụ:
- "Sữa tươi Vinamilk" vs "Sữa tươi TH True Milk" → overlap = 2/2 = 1.0
- "Sữa tươi Vinamilk có đường" vs "Sữa tươi" → overlap = 2/2 = 1.0

##### Ensemble cuối cùng

```python
final_similarity = alpha * Overlap_Score(Count) + (1 - alpha) * Cosine(TF-IDF)
# alpha = CB_ALPHA = 0.5
```

#### 4.1.2 Tiền xử lý văn bản (Text Preprocessing)

`vectorizer.py` sử dụng **`_clean_text_preprocessor`** — hàm tiền xử lý cực nhanh nhờ:

1. **Lowercase** toàn bộ
2. **Xóa số đo/dung tích** bằng regex: `\b\d+(?:[\.,]\d+)?\s*(?:ct|count|mg|mcg|oz|fl\s*oz|fl|gallon|inch|in|pack|pk|ml|liter|lit|lít|l|lb|lbs|iu|i\.u\.?|loads|watt|cups|cup|sticks|g|kg|gr|grs|cm|mm)\b`
3. **Xóa stopwords tiếng Việt** từ file `vietnamese_stopwords.txt` — dùng 2 regex đã compiled sẵn (1 cho word, 1 cho ký tự đặc biệt), khởi tạo 1 lần ở global scope
4. **Xóa ký tự đặc biệt** còn sót lại: `[^\w\s]`
5. **Dọn khoảng trắng thừa**

#### 4.1.3 Cách hoạt động — Inference-time (on-demand)

Thay vì pre-compute toàn bộ ma trận similarity 36K × 36K (tốn ~8GB RAM), CB chỉ tính similarity **on-demand** cho các cặp (A, B) mà co-occurrence models thực sự đề xuất:

```python
# CBFilter.filter(product_a_id, candidates, threshold)
# 1. Map product_id → index trong ma trận vector
idx_a = product_id_to_idx[product_a_id]
valid_indices = [product_id_to_idx[cid] for cid in candidates]

# 2. Tính ensemble similarity on-demand
similarities = cb_ensemble_similarity(
    tfidf_vectors, count_vectors,
    idx_a, valid_indices, alpha=0.5
)

# 3. Lọc:
# - similarity >= threshold → substitute → LOẠI BỎ
# - similarity < threshold  → complementary → GIỮ LẠI
# - similarity == 0         → hoàn toàn khác → complementary mạnh → GIỮ LẠI
mask = similarities < threshold  # ENS_CB_THRESHOLD = 0.25
```

#### 4.1.4 Xử lý Cold-start

Nếu sản phẩm đầu vào hoặc candidate không có vector (không nằm trong `product_id_to_idx`):
- **Sản phẩm đầu vào cold-start**: Giữ nguyên toàn bộ candidates (không filter)
- **Candidate cold-start**: Giữ lại candidate đó (bỏ qua filter)

#### 4.1.5 Ưu điểm & Nhược điểm

**Ưu điểm:**
- Chỉ tính similarity khi cần, không lưu full matrix → tiết kiệm bộ nhớ
- Dùng tên + aisle + department tiếng Việt → phát hiện substitute ngay cả khi chưa có dữ liệu mua hàng
- Ensemble Count + TF-IDF: Count bắt từ trùng tuyệt đối, TF-IDF phân hóa từ khóa chính/phụ
- Regex stopwords compiled sẵn → tiền xử lý cực nhanh

**Nhược điểm:**
- Chỉ dựa trên thông tin mô tả, không capture hành vi mua hàng thực tế
- Similarity = 0 (tên hoàn toàn khác) không có thông tin → không thể kết luận

---

### 4.2 Item-Based Collaborative Filtering (Item-CF) — Ochiai + Confidence Score

**File code**: `src/models/item_cf.py` — class `ItemCFModel`

**Ý tưởng**: Đếm số lần cặp sản phẩm (A, B) cùng xuất hiện trong một đơn hàng, sau đó tính score phản ánh cả **mức độ liên kết** lẫn **hướng gợi ý**.

#### 4.2.1 Thuật toán chi tiết

##### Bước 1: Xây CSR co-occurrence matrix (dùng Numba JIT)

Dữ liệu 31.9M records, 36K products → cần xử lý hiệu quả:

```python
# 1a. Nhóm order_products theo order_id → list các product indices
grouped = order_products.groupby('order_id')['product_id']

# 1b. Xây CSR representation:
#     order_indices: flat array [p1, p2, p3, ...] — tất cả products trong tất cả orders
#     order_ptr:     [0, len(order1), len(order1+order2), ...] — start/end của mỗi order
order_indices = np.array([...])  # tổng items ≈ 31.9M
order_ptr = np.array([0, len1, len1+len2, ...])  # n_orders+1 ≈ 3.3M+1

# 1c. Đếm co-occurrence pairs bằng Numba (count_pairs_numba)
rows, cols, counts = count_pairs_numba(order_indices, order_ptr, n_products)
# rows[i], cols[i] = cặp sản phẩm (idx_a, idx_b)
# counts[i] = số lần xuất hiện cùng nhau
```

Hàm `count_pairs_numba` (JIT-compiled trong `_numba_ops.py`):
- Phase 1: Duyệt orders → đếm tổng pairs upper bound
- Phase 2: Ghi pairs vào arrays (tránh a==b)
- Sort + Reduce duplicates → unique pairs với count

##### Bước 2: Lọc min_support

```python
MIN_SUPPORT = 10  # pair xuất hiện < 10 lần → noise thống kê → loại bỏ
mask = counts >= MIN_SUPPORT
rows, cols, counts = rows[mask], cols[mask], counts[mask]
```

Lý do: pair xuất hiện < 10 lần trên 31.9M giao dịch là noise, không đáng tin.

##### Bước 3: Xây CSR matrix đối xứng

```python
# Thêm cả (b, a) để ma trận đối xứng — truy vấn nhanh
rows_all = np.concatenate([rows, cols])
cols_all = np.concatenate([cols, rows])
data_all = np.concatenate([counts, counts])

cooc_matrix = sparse.csr_matrix(
    (data_all, (rows_all, cols_all)),
    shape=(n_products, n_products),
    dtype=np.int32
)
```

##### Bước 4: Tính product counts

```python
product_counts = np.bincount(order_indices, minlength=n_products)
# product_counts[i] = số lần sản phẩm i xuất hiện trong tất cả orders
```

##### Bước 5: Score formula (tại inference)

```python
def _compute_scores(product_idx):
    cnt_i = product_counts[product_idx]          # count(A)
    row = cooc_matrix[product_idx].toarray()[0]  # cnt(A, B) với mọi B
    
    # Ochiai coefficient = Cosine similarity trên binary vector
    ochiai = cnt / sqrt(cnt_i * cnt_j)           # cho mỗi cặp (A, B)
    
    # Confidence (bất đối xứng, có hướng)
    conf = cnt / cnt_i                           # P(B|A) = cnt(A,B) / count(A)
    
    # Log frequency (popularity bonus)
    log_freq = log1p(cnt)                        # log(1 + cnt)
    
    # Score cuối — có hướng
    score(A → B) = ochiai × conf × log_freq
```

#### 4.2.2 Giải thích trực quan

| Thành phần | Công thức | Tính chất | Ý nghĩa |
|-----------|-----------|----------|---------|
| **Ochiai** | `cnt / sqrt(count(A) × count(B))` | Đối xứng | Normalize theo độ phổ biến cả 2 bên, không phạt item phổ biến như PMI |
| **Confidence** | `cnt / count(A)` | Bất đối xứng | "Nếu mua A, khả năng mua B là bao nhiêu?" — cho hướng gợi ý |
| **Log frequency** | `log(1 + cnt)` | Đối xứng | Thưởng cho pair có volume lớn, tránh overfit pair hiếm |

Ví dụ:
- "Mua **nước lẩu** → gợi ý **lẩu**": conf cao vì ai mua nước lẩu cũng mua lẩu
- "Mua **lẩu** → gợi ý **nước lẩu**": conf thấp hơn vì không phải ai mua lẩu cũng mua nước lẩu đó

#### 4.2.3 Tại sao không dùng PMI / SPPMI?

| Vấn đề | PMI/SPPMI | Item-CF (Ochiai + Conf) |
|---|---|---|
| Item phổ biến bị underrate (cơm + gà) | ❌ PMI phạt nặng P(A) lớn | ✅ Ochiai normalize cân bằng (Item-Based CF) |
| Pair hiếm 1–2 lần bị overrate | ⚠️ SPPMI giảm nhưng chưa đủ | ✅ log1p suppress + min_support |
| Chỉ cho score đối xứng | ✅/❌ | ✅ Confidence có hướng |
| Substitute detection qua PMI âm | ✅ | ❌ Nhưng CB filter đã lo phần này |

---

### 4.3 Item2Vec (Neural Item-Based CF) — Word2Vec Skip-gram trên Giỏ Hàng

**File code**: `src/models/item_cf_neural.py` — class `ItemCFNeuralModel`

**Ý tưởng**: Coi mỗi đơn hàng (order) như một **"câu"**, mỗi sản phẩm (product) là một **"từ"**. Áp dụng Word2Vec để học embedding cho từng sản phẩm từ ngữ cảnh mua hàng.

#### 4.3.1 Cách hoạt động

```python
# 1. Mỗi order → một "câu" (list string product_id)
sentences = []
for order_id, group in grouped:
    items = [str(pid) for pid in group if pid in product_id_to_idx]
    if len(items) >= 2:  # Order phải có ít nhất 2 items
        sentences.append(items)
# sentences ≈ 3.3M câu

# 2. Train Word2Vec Skip-gram
model = Word2Vec(
    sentences=sentences,
    vector_size=128,        # embedding dimension
    window=5,               # context window size
    min_count=10,           # bỏ qua sản phẩm xuất hiện < 10 lần
    negative=10,            # số negative samples
    epochs=20,              # số lần lặp
    workers=4,              # parallel threads
    sg=1,                   # Skip-gram (1) vs CBOW (0)
    seed=42,
)
```

#### 4.3.2 Hyperparameters

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `vector_size` | 128 | Số chiều embedding |
| `window` | 5 | Số sản phẩm lân cận trong order |
| `min_count` | 10 | Sản phẩm xuất hiện ít hơn sẽ bị bỏ qua |
| `negative` | 10 | Số negative samples cho Skip-gram |
| `epochs` | 20 | Số lần lặp qua toàn bộ dữ liệu |
| `workers` | 4 | Số luồng song song |

#### 4.3.3 Gợi ý (Recommendation)

```python
def recommend(product_id, top_k=100):
    pid_str = str(product_id)
    similar = model.wv.most_similar(pid_str, topn=top_k)
    return [(int(pid), float(sim)) for pid, sim in similar]
```

#### 4.3.4 Ưu điểm & Nhược điểm

**Ưu điểm:**
- Học được **ngữ nghĩa sản phẩm từ ngữ cảnh mua hàng** — không chỉ đếm cặp trực tiếp
- Capture được **mối quan hệ gián tiếp**: A hay đi với B, B hay đi với C → A có liên quan đến C
- Embedding có thể tái sử dụng cho nhiều downstream tasks

**Nhược điểm:**
- Cần tuning nhiều hyperparameter
- Item2Vec không phân biệt được hướng (đối xứng, không như Item-CF có hướng)
- Sản phẩm có `count < min_count` sẽ không có embedding (long-tail issue)

---

### 4.4 Graph-based: KGMetapath (KG Embedding + Metapath Walk)

**File code**: `src/models/kg_metapath.py` — class `KGMetapathModel`

**Ý tưởng**: Xây dựng **Đồ thị Tri thức đa thể (Heterogeneous Knowledge Graph - KG)** với 3 loại nút (Product, Aisle, Department) và 3 loại quan hệ (CO_OCCUR, BELONGS_TO, PART_OF). Sử dụng **Metapath Walk** — duyệt đồ thị định hướng ngữ nghĩa — thay vì uniform random walk để giải quyết bài toán siêu nút (supernode) và long-tail.

#### 4.4.1 Cấu trúc KG

```
Product (P)     — 36,181 nodes (đã loại non-food)
Aisle (A)       — 134 nodes  
Department (D)  — 21 nodes

Cạnh:
- P1 --CO_OCCUR--> P2  (đồng xuất hiện ≥ threshold, từ ma trận co-occurrence)
- P  --BELONGS_TO--> A   (quan hệ danh mục)
- A  --PART_OF--> D      (quan hệ phân cấp)
```

#### 4.4.2 Xây dựng KG — 4 bước

##### Bước 1: Mapping Product → Aisle

```python
for _, row in products_df.iterrows():
    pid = row['product_id']
    aid = row['aisle_id']
    if pid in product_id_to_idx:
        pidx = product_id_to_idx[pid]
        product_to_aisle[pidx] = aid
        aisle_to_products.setdefault(aid, []).append(pidx)
```

##### Bước 2: Đếm CO_OCCUR edges (dùng Numba)

Giống Item-CF: dùng `count_pairs_numba` trên 31.9M records → pairs (A, B) + count.

Lọc `edge_threshold=10`: chỉ giữ pairs xuất hiện ≥ 10 lần.

##### Bước 3: Xây adjacency CSR (dùng Numba)

```python
indptr, neighbors, weights = _build_adjacency_csr(
    pair_rows, pair_cols, pair_counts, n_products
)
# indptr:    [0, deg1, deg1+deg2, ...] — start/end mỗi node
# neighbors: flat array các neighbor
# weights:   flat array co-occurrence counts
```

Hàm `_build_adjacency_csr` (JIT-compiled):
- Đếm degree mỗi node
- Build indptr
- Fill neighbors + weights (cả 2 hướng)
- Sort neighbors mỗi node để dùng binary search

##### Bước 4: Xây graph dict cho Python access

```python
graph = {}
for node in range(n_products):
    start, end = indptr[node], indptr[node+1]
    if start < end:
        graph[node] = [(neighbors[i], weights[i]) for i in range(start, end)]
```

#### 4.4.3 Metapath Walk — 2 kịch bản

Tỉ lệ: **50% Behavioral, 50% Semantic** (tham số `MW_METAPATH_BEHAVIORAL_RATIO = 0.5`)

##### Kịch bản 1 — Behavioral (học hành vi đồng xuất hiện):

```
P1 --CO_OCCUR--> P2 --CO_OCCUR--> P3
```

Robot đi theo các cạnh CO_OCCUR, bắt trọn vẹn các mẫu kết hợp giỏ hàng thực tế.

```python
# Chọn neighbor ngẫu nhiên trong danh sách neighbor của node hiện tại
s, e = indptr[cur], indptr[cur+1]
n_nb = e - s
nb_idx = rand_matrix[walk_idx, step] % n_nb
cur = neighbors[s + nb_idx]
```

##### Kịch bản 2 — Semantic (cứu vớt hàng hiếm đuôi dài):

```
P1 --BELONGS_TO--> A --random--> P2
```

Robot thoát khỏi vòng lặp sản phẩm phổ biến thông qua nút Aisle, tiếp cận các sản phẩm hiếm (46.3% sản phẩm có dưới 50 lần tương tác).

```python
# Bước 1: từ P đi lên Aisle
aisle_id = product_to_aisle[cur]

# Bước 2: từ Aisle chọn sản phẩm khác
candidates = [p for p in products_in_aisle if p != cur]
cur = candidates[rand_matrix[walk_idx, step] % len(candidates)]
```

##### Fallback mechanism

Nếu Behavioral walk gặp node không có neighbor CO_OCCUR → reset về start node (semantic). Nếu Semantic walk gặp node không có aisle → fallback về Behavioral. Điều này đảm bảo walk không bị đứt giữa chừng.

#### 4.4.4 Tối ưu hiệu năng Walk

- **Pre-generate random numbers**: `rand_matrix = np.random.randint(0, max_degree, size=(total_walks, walk_length))` — tránh gọi `np.random` nhiều lần trong vòng lặp
- **Pre-allocate walks array**: `walks_arr = np.full((total_walks, walk_length), -1, dtype=np.int32)` — tránh append liên tục
- **Track walk_lengths**: mỗi walk có độ dài thực tế khác nhau (có thể bị đứt sớm)

#### 4.4.5 Train Word2Vec trên Metapath Walks

```python
# Convert walks → sentences
sentences = []
for i in range(walks.shape[0]):
    wlen = walk_lengths[i]
    if wlen > 1:
        sentence = [str(walks[i, j]) for j in range(wlen)]
        sentences.append(sentence)

# Train Skip-gram
model = Word2Vec(
    sentences=sentences,
    vector_size=128,
    window=10,        # context window cho walk sequences
    min_count=1,      # giữ tất cả product nodes
    negative=10,
    epochs=20,
    workers=4,
    sg=1,
)
```

Extract embeddings: `self.embeddings[node] = model.wv[pid_str]` — caching norms cho cosine similarity nhanh.

#### 4.4.6 Gợi ý (Recommendation)

```python
def recommend(product_id, top_k=100):
    idx = product_id_to_idx[product_id]
    vec_a = embeddings[idx]
    norm_a = np.linalg.norm(vec_a)
    
    # Cosine similarity với tất cả nodes (vectorized)
    similarities = (embeddings @ vec_a) / (embedding_norms * norm_a)
    similarities[idx] = -1  # bỏ qua chính nó
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(idx_to_product_id[i], similarities[i]) for i in top_indices if similarities[i] > 0]
```

#### 4.4.7 Ưu điểm & Nhược điểm

**Ưu điểm:**
- **Giải quyết triệt để supernode**: Không bị kẹt ở Aisle/Department nhờ lộ trình cố định
- **Xử lý long-tail**: Sản phẩm hiếm vẫn được kết nối không gian qua quan hệ danh mục
- **High-order relationships**: Giữ được ưu điểm của graph embedding
- **White-box walk**: Kiểm soát minh bạch từng bước toán học

**Nhược điểm:**
- Phức tạp hơn DeepWalk trong việc xây dựng đồ thị đa thể
- Cần tuning thêm tham số `behavioral_ratio`

---

### 4.5 Hybrid Ensemble (Kết hợp Co-occurrence Models)

**File code**: `src/models/ensemble.py` — class `EnsembleModel`

**Ý tưởng**: Mỗi model co-occurrence capture một khía cạnh khác nhau của mối quan hệ sản phẩm. Kết hợp chúng bằng weighted score để tận dụng điểm mạnh của từng model.

#### 4.5.1 Công thức ensemble

```python
final_score(A → B) = α × ItemCF_score(A, B)
                   + β × Item2Vec_sim(A, B)
                   + γ × KGMetapath_sim(A, B)
```

Trong đó:
- `α = 0.5, β = 0.25, γ = 0.25` (tổng = 1.0)
- Mỗi score được **normalize về [0, 1]** trước khi kết hợp để tránh một model dominate

**Đóng góp của từng model:**

| Model | α/β/γ | Khía cạnh đóng góp |
|---|---|---|
| **Item-CF (Ochiai + Conf)** | 0.5 | Mối quan hệ đồng xuất hiện trực tiếp, có hướng, robust với item phổ biến |
| **Item2Vec** | 0.25 | Ngữ nghĩa sản phẩm từ context mua hàng, mối quan hệ gián tiếp |
| **KGMetapath** | 0.25 | Mối quan hệ bậc cao trong cấu trúc đồ thị KG, xử lý long-tail nhờ semantic walk |

> **CB (TF-IDF) không tham gia ensemble** — CB là tầng hậu xử lý độc lập, áp dụng sau khi có kết quả ensemble.

#### 4.5.2 Quy trình recommend

```python
def recommend(product_id, use_cb_filter=True, top_k=10):
    # 1. Lấy top-K candidate từ mỗi model
    item_cf_recs = item_cf.recommend(product_id, top_k=100)
    i2v_recs = item2vec.recommend(product_id, top_k=100)
    mw_recs = metapath2vec.recommend(product_id, top_k=100)
    
    # 2. Union các candidate
    candidate_ids = set(pid for pid, _ in item_cf_recs)
    candidate_ids |= set(pid for pid, _ in i2v_recs)
    candidate_ids |= set(pid for pid, _ in mw_recs)
    
    # 3. Tính weighted score (sau min-max normalize)
    item_cf_norm = normalize([item_cf_dict.get(pid, 0) for pid in candidates])
    i2v_norm = normalize([i2v_dict.get(pid, 0) for pid in candidates])
    mw_norm = normalize([mw_dict.get(pid, 0) for pid in candidates])
    
    final_scores = [α*icf + β*i2v + γ*mw for icf, i2v, mw in zip(...)]
    
    # 4. Sort descending
    candidates_sorted = sorted(zip(candidates, final_scores), key=lambda x: -x[1])
    
    # 5. (Optional) CB Filter loại substitute
    if use_cb_filter and cb_filter is not None:
        candidates_sorted = cb_filter.filter(
            product_id, candidates_sorted, threshold=ENS_CB_THRESHOLD
        )
    
    # 6. Trả về top-K
    return candidates_sorted[:top_k]
```

#### 4.5.3 Normalize trước khi ensemble

```python
def _normalize(scores):
    """Min-max normalization về [0, 1]."""
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s + 1e-9) for s in scores]
```

#### 4.5.4 Save / Load

Ensemble lưu:
- **config.json**: α, β, γ, top_k, final_k
- **cb_tfidf_vectors.npz**: TF-IDF vectors (CSR sparse)
- **cb_count_vectors.npz**: Count vectors (CSR sparse)
- **product_id_to_idx.json**: mapping

Các sub-models (Item-CF, Item2Vec, KGMetapath) được lưu riêng ở steps 03, 04, 05 và load lại khi cần.

---

### 4.6 Tiêu Chí So Sánh

| Tiêu chí | Item-CF | Item2Vec | KGMetapath | CB Filter | Ensemble (w/o CB) | Ensemble + CB |
|---|---|---|---|---|---|---|
| **Loại CF** | Memory-based Item-CF ✅ | Neural Item-Based CF ✅ | Không phải CF (Graph-based) ❌ | — | — | — |
| **Vai trò** | Model gợi ý | Model gợi ý | Model gợi ý | Bộ lọc hậu xử lý | Ensemble thuần (ICF+I2V+MW) | Ensemble + Filter |
| **LLM Eval (Precision@10)** | ? | ? | ? | — ⚠️ | ? | ? (kỳ vọng cao nhất) |
| **Loại bỏ substitute** | ❌ | ❌ | ❌ | ✅ Tốt | ❌ | ✅ Tốt |
| **Item phổ biến (cơm+gà)** | ✅ Item-CF fix | ✅ | ✅ | — | ✅ | ✅ |
| **Pair hiếm / noise** | ✅ min_support=10 | ⚠️ min_count=10 | ⚠️ threshold=10 | — | ✅ | ✅ |
| **Cold-start** | ❌ | ❌ | ⚠️ (qua aisle) | ✅ (filter) | ❌ | ✅ |
| **Long-tail** | ⚠️ | ⚠️ | ✅ (semantic walk) | ✅ (filter) | ⚠️ | ✅ |
| **Hướng gợi ý (A→B ≠ B→A)** | ✅ | ❌ | ❌ | — | ✅ (từ Item-CF) | ✅ (từ Item-CF) |
| **Khả năng giải thích** | Cao | TB | Thấp | Cao | TB | TB |
| **Thời gian train** | Nhanh | TB | Chậm | Nhanh | TB | TB |

> ⚠️ **CB Filter không có LLM Eval riêng**: CB là deterministic filter, tác động của nó được đánh giá **gián tiếp** qua việc so sánh `Ensemble (w/o CB)` vs `Ensemble + CB`. Nếu Ensemble+CB có Precision@10 cao hơn, chứng tỏ CB đang loại bỏ substitute hiệu quả.

---

### 4.7 Thứ Tự Triển Khai

| Bước | Script | Việc cần làm | Lý do |
|------|--------|-------------|-------|
| **1** | `model/01_load_data.py` | Load & lọc dữ liệu (loại non-food) | Tiền đề cho tất cả các bước sau |
| **2** | `model/02_cb_filter.py` | Vector hóa sản phẩm (TF-IDF + Count multi-field) | Cần xong trước để dùng làm filter ở bước 6 |
| **3** | `model/03_item_cf.py` | Item-CF (Item-Based CF) | Nhanh, hiệu quả, là backbone của ensemble |
| **4** | `model/04_item_cf_neural.py` | Item2Vec | Bổ sung ngữ nghĩa cho ensemble |
| **5** | `model/05_kg_metapath.py` | KGMetapath | Xây KG + Metapath Walk, bổ sung high-order relationships + xử lý long-tail |
| **6** | `model/06_ensemble.py` | Co-occurrence Ensemble + CB Filter | Kết hợp bước 3+4+5, thêm CB filter loại substitute |
| **7** | `model/07_eval_llm.py` | LLM Evaluation | Đánh giá tất cả model bằng ground truth từ LLM |

Mỗi bước đều có thể **chạy độc lập** và **đánh giá riêng** qua LLM evaluation (xem **Mục 5**) trước khi tiến sang bước tiếp theo.

---

## 5. Survey Dataset — Ground Truth cho Human Evaluation bằng LLM

### 5.1 Tổng quan

Sau khi train xong tất cả các model, cần một tập **ground truth** để đánh giá định lượng chất lượng gợi ý mua kèm. Do bài toán gợi ý mua kèm là **phi tuần tự** và không có sẵn nhãn từ dữ liệu Instacart gốc, tập ground truth này được xây dựng thông qua khảo sát sử dụng **Model Ngôn Ngữ Lớn (LLM)** làm người đánh giá.

LLM đóng vai trò như một "người dùng thông thái" — dựa trên kiến thức về sản phẩm và thói quen mua sắm, LLM đưa ra nhận xét liệu một cặp sản phẩm có thực sự **mua kèm (complementary)** hay không.

### 5.2 Phương pháp thu thập

#### 5.2.1 Cấu trúc mẫu khảo sát

Mỗi mẫu khảo sát là một cặp `(product_A, product_B)`. LLM được hỏi: *"Nếu khách hàng mua product_A, liệu họ có khả năng mua product_B trong cùng giỏ hàng không?"* — trả lời **Có (1)** hoặc **Không (0)**.

Hai chiến lược lấy mẫu:

| Loại mẫu | Tỷ lệ | Mô tả |
|---|---|---|
| **Top-5** | 50% | Chọn `product_A` ngẫu nhiên, lấy `product_B` từ top-5 gợi ý của model |
| **Nhiễu (Noise)** | 50% | Chọn `product_A` ngẫu nhiên, lấy `product_B` ngẫu nhiên từ các sản phẩm **không nằm** trong top-K gợi ý của model |

Lý do 50-50: Top-5 đo khả năng model gợi ý đúng, nhiễu đo khả năng model **không** gợi ý sai.

Tất cả các model được đánh giá trên **cùng một bộ mẫu** (cùng seed ngẫu nhiên) để đảm bảo công bằng.

#### 5.2.2 LLM làm người đánh giá

- Sử dụng **Gemini 2.0 Flash** (qua API)
- LLM được cung cấp tên sản phẩm (`product_name`), có thể kèm aisle, department để có đủ ngữ cảnh
- Mỗi cặp được hỏi độc lập, không tiết lộ model nào tạo ra mẫu đó
- LLM trả về danh sách các cặp complementary kèm mô tả món ăn tạo thành

### 5.3 Cấu trúc dữ liệu (Schema)

Dữ liệu khảo sát được lưu trong thư mục `data/survey/`, gồm các file sau:

#### 5.3.1 File input: `survey_samples.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `product_A_id` | int | ID sản phẩm đầu vào (target) |
| `product_A_name` | str | Tên sản phẩm A (tiếng Việt) |
| `product_B_id` | int | ID sản phẩm ứng viên |
| `product_B_name` | str | Tên sản phẩm B (tiếng Việt) |

Mỗi dòng là một cặp `(A → B)` từ union top-10 của 5 model (Item-CF, Item2Vec, KGMetapath, Ensemble, Ensemble+CB).

#### 5.3.2 File raw LLM response: `llm_raw_responses/gemini_responses.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `product_A_id` | int | ID sản phẩm đầu vào (target) |
| `product_B_id` | int | ID sản phẩm ứng viên được LLM chọn |
| `description` | str | Món ăn tạo thành khi kết hợp A+B |

File này **chỉ chứa các cặp complementary** (llm_label=1). Các cặp trong `survey_samples.csv` không xuất hiện trong file này mặc định có `llm_label = 0`.

#### 5.3.3 File output: `survey_labeled.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `product_A_id` | int | ID sản phẩm đầu vào |
| `product_A_name` | str | Tên sản phẩm A |
| `product_B_id` | int | ID sản phẩm ứng viên |
| `product_B_name` | str | Tên sản phẩm B |
| `llm_label` | int | 1 = complementary, 0 = not complementary |
| `description` | str | Món ăn tạo thành khi kết hợp A+B (rỗng nếu llm_label=0) |

### 5.4 Pipeline xử lý (script 07_eval_llm)

```
survey_samples.csv (4 cột — union top-10 từ 5 model)
        │
        ▼
Gửi cho LLM (Gemini 2.0 Flash) theo prompt:
  "Với sản phẩm A, trong danh sách B dưới đây,
   chọn các B thực sự mua kèm để tạo thành món ăn"
        │
        ▼
llm_raw_responses/gemini_responses.csv
  (CSV: product_A_id,product_B_id,description — chỉ chứa các cặp complementary)
        │
        ▼
Gộp với survey_samples.csv → survey_labeled.csv
  (llm_label=1 nếu có trong gemini_responses, ngược lại =0)
        │
        ▼
Tính metrics cho từng model:
  Precision@10, Recall@10, F1@10, Hit@10
```

### 5.5 Các chỉ số đánh giá (Metrics)

Tất cả các chỉ số được tính trên top-10 gợi ý của mỗi model, sử dụng ground truth từ LLM.

#### 5.5.1 Precision@10

```python
Precision@10 = số lượng gợi ý đúng (complementary) trong top-10 / 10
```

**Ý nghĩa**: Trong số 10 sản phẩm được gợi ý, có bao nhiêu sản phẩm thực sự là mua kèm?

#### 5.5.2 Recall@10

```python
Recall@10 = số lượng gợi ý đúng trong top-10 / tổng số sản phẩm complementary (trong ground truth)
```

**Ý nghĩa**: Model đã gợi ý được bao nhiêu phần trăm sản phẩm mua kèm thực sự?

#### 5.5.3 F1@10

```python
F1@10 = 2 × (Precision@10 × Recall@10) / (Precision@10 + Recall@10)
```

**Ý nghĩa**: Trung bình điều hòa giữa Precision và Recall — đánh giá tổng thể.

#### 5.5.4 Hit@10

```python
Hit@10 = 1 nếu có ít nhất 1 gợi ý đúng trong top-10, ngược lại = 0
```

**Ý nghĩa**: Model có gợi ý được ít nhất một sản phẩm mua kèm hữu ích hay không?

### 5.6 Cách dùng để so sánh model

Mỗi model được đánh giá trên cả 4 chỉ số. Kết quả được tổng hợp trong bảng so sánh:

| Model | Precision@10 | Recall@10 | F1@10 | Hit@10 |
|---|---|---|---|---|
| Item-CF (Item-Based CF) | ⬜ | ⬜ | ⬜ | ⬜ |
| Item2Vec | ⬜ | ⬜ | ⬜ | ⬜ |
| KGMetapath | ⬜ | ⬜ | ⬜ | ⬜ |
| Ensemble (w/o CB) | ⬜ | ⬜ | ⬜ | ⬜ |
| **Ensemble + CB Filter** | ⬜ | ⬜ | ⬜ | ⬜ |

> **Kỳ vọng**: Ensemble + CB Filter đạt điểm cao nhất ở tất cả các chỉ số nhờ kết hợp sức mạnh của nhiều model và bộ lọc substitute. So sánh `Ensemble (w/o CB)` vs `Ensemble + CB` cho thấy tác động của CB filter trong việc loại bỏ substitute và cải thiện chất lượng gợi ý.