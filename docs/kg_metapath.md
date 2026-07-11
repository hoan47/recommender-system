# KG Metapath — Giải Thích Chi Tiết Từ A-Z

> Tài liệu này giải thích mô hình KG Metapath (Knowledge Graph Metapath) trong dự án, từ cách xây dựng đồ thị tri thức, sinh metapath walks, đến training Word2Vec để học embedding. Phần Word2Vec (Item2Vec) sẽ được chỉ dẫn quay lại file riêng.

---

## Mục lục

1. [KG Metapath là gì?](#1-kg-metapath-là-gì)
2. [Tổng quan luồng hoạt động](#2-tổng-quan-luồng-hoạt-động)
3. [Bước 1: Xây dựng Knowledge Graph (KG)](#3-bước-1-xây-dựng-knowledge-graph-kg)
4. [Bước 2: Co-occurrence Counting](#4-bước-2-co-occurrence-counting)
5. [Bước 3: Metapath Walk — 2 kịch bản](#5-bước-3-metapath-walk--2-kịch-bản)
6. [Bước 4: Word2Vec (Skip-gram + Negative Sampling)](#6-bước-4-word2vec-skip-gram--negative-sampling)
7. [Bước 5: Recommend bằng Cosine Similarity](#7-bước-5-recommend-bằng-cosine-similarity)
8. [Ví dụ chi tiết bằng số](#8-ví-dụ-chi-tiết-bằng-số)
9. [KG Metapath vs Item-CF (Ochiai) vs Item2Vec thuần](#9-kg-metapath-vs-item-cf-ochiai-vs-item2vec-thuần)
10. [Ưu điểm & Nhược điểm](#10-ưu-điểm--nhược-điểm)

---

## 1. KG Metapath là gì?

**KG Metapath** là phương pháp kết hợp **Knowledge Graph (KG)** và **Metapath Walk** để học embedding cho sản phẩm.

**Ý tưởng chính:**
1. Xây dựng đồ thị tri thức với nhiều loại nút (Product, Aisle, Department)
2. Sinh các đường đi (walks) trên đồ thị theo các kịch bản khác nhau
3. Dùng **Word2Vec (Item2Vec)** để học embedding từ các walks đó

**Điểm khác biệt so với Item2Vec thuần:**
- Item2Vec thuần: "câu" = đơn hàng thật
- KG Metapath: "câu" = các đường đi được sinh từ KG

---

## 2. Tổng quan luồng hoạt động

```
DỮ LIỆU: order_products.parquet (31.9M records)
         products.parquet (36,181 records)
    │
    ▼
┌─────────────────────────────────────────┐
│  Bước 1: Xây dựng Knowledge Graph       │
│  - 3 loại nút: Product, Aisle, Dept     │
│  - 3 loại cạnh: CO_OCCUR, BELONGS_TO,   │
│    PART_OF                               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Bước 2: Đếm co-occurrence pairs        │
│  (giống Item-CF) → lọc threshold        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Bước 3: Metapath Walk                  │
│  - Kịch bản 1: Behavioral (P→P→P)      │
│  - Kịch bản 2: Semantic (P→A→P)        │
│  → Sinh ra các "câu" (sequences)        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Bước 4: Word2Vec (Skip-gram)           │
│  → Mỗi sản phẩm có 1 embedding vector   │
│  → Chi tiết xem docs/item2vec.md        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Bước 5: Recommend                      │
│  cosine(embedding_A, embedding_B)       │
└─────────────────────────────────────────┘
```

---

## 3. Bước 1: Xây dựng Knowledge Graph (KG)

### 3.1. Các loại nút (Nodes)

KG có 3 loại nút:

| Loại nút | Số lượng | Ý nghĩa | Ví dụ |
|----------|----------|---------|-------|
| **Product (P)** | ~36,181 | Sản phẩm | Sữa, Bánh mì, Trứng |
| **Aisle (A)** | 134 | Gian hàng | Sữa tươi, Bánh mì tươi |
| **Department (D)** | 21 | Khu vực | Thực phẩm, Đồ uống |

### 3.2. Các loại cạnh (Edges)

| Cạnh | Từ → Đến | Ý nghĩa | Trọng số |
|------|---------|---------|----------|
| **CO_OCCUR** | P ↔ P | 2 sản phẩm hay mua chung | Số lần co-occurrence |
| **BELONGS_TO** | P → A | Sản phẩm thuộc gian hàng nào | 1 (cố định) |
| **PART_OF** | A → D | Gian hàng thuộc khu vực nào | 1 (cố định) |

### 3.3. Ví dụ KG

```
         ┌─── Department ───┐
         │  Thực phẩm (D1)  │
         └────────┬─────────┘
                  │ PART_OF
         ┌────────▼─────────┐
         │  Aisle: Sữa (A1) │
         └──┬──────────┬────┘
            │BELONGS_TO│BELONGS_TO
       ┌────▼───┐  ┌───▼────┐
       │Sữa (P1)│  │Sữa chua│
       └───┬────┘  │(P3)    │
           │       └────────┘
           │ CO_OCCUR
       ┌───▼────┐
       │Bánh mì │
       │(P2)    │
       └────────┘
```

---

## 4. Bước 2: Co-occurrence Counting

### 4.1. Giống hệt Item-CF

Đây là bước giống Item-CF (Ochiai) ở bước 3 — đếm số lần 2 sản phẩm xuất hiện cùng đơn hàng.

```python
# Từ order_products, đếm cặp (A, B) xuất hiện cùng order
rows, cols, counts = count_pairs_numba(order_indices, order_ptr, n_products)
```

### 4.2. Lọc edge threshold

Khác với Item-CF dùng `min_support` (mặc định 2), KG dùng `edge_threshold` (MW_EDGE_THRESHOLD) — thường cao hơn để đồ thị bớt dày đặc:

```python
mask = counts >= self.params['edge_threshold']  # VD: threshold = 3
pair_rows = rows[mask]
pair_cols = cols[mask]
pair_counts = counts[mask]
```

### 4.3. Xây adjacency list

Dùng Numba để xây CSR adjacency cho graph:

```python
indptr, neighbors, weights = _build_adjacency_csr(
    pair_rows, pair_cols, pair_counts, n_products
)
```

Kết quả: mỗi node có danh sách `(neighbor, weight)` — weight = số lần co-occurrence.

---

## 5. Bước 3: Metapath Walk — 2 kịch bản

### 5.1. Tổng quan

**Metapath Walk** = đi dạo trên đồ thị KG để sinh ra các "câu" (sequences of product IDs). Mỗi "câu" này sẽ được đưa vào Word2Vec để học embedding.

**Kích thước walks:**
- `num_walks` = MW_NUM_WALKS (VD: 5) — số lần đi từ mỗi node
- `walk_length` = MW_WALK_LENGTH (VD: 20) — độ dài mỗi bước đi
- Tổng walks ≈ số nodes × num_walks ≈ 36,181 × 5 ≈ 180,905 walks

### 5.2. Kịch bản 1: Behavioral Walk (P → P → P ...)

**Ý tưởng:** Đi theo các cạnh CO_OCCUR — 2 sản phẩm hay mua chung.

**Cách đi:**
```
Bắt đầu từ node X
  Bước 1: chọn ngẫu nhiên 1 neighbor của X (theo trọng số)
  Bước 2: từ node đó, chọn tiếp 1 neighbor của nó
  ... lặp cho đến walk_length
```

**Ví dụ:**
```
Sữa → Bánh_mì → Trứng → Bơ → Sữa_chua → ...
```

**Khi nào dùng?** `behavioral_ratio` = MW_METAPATH_BEHAVIORAL_RATIO (VD: 0.7) — 70% walks là behavioral.

### 5.3. Kịch bản 2: Semantic Walk (P → A → P)

**Ý tưởng:** Nhảy qua Aisle để tiếp cận sản phẩm cùng gian hàng — giúp học được quan hệ "cùng loại".

**Cách đi:**
```
Bước 1: Từ Product X, đi lên Aisle của nó (BELONGS_TO)
Bước 2: Từ Aisle, chọn ngẫu nhiên 1 Product Y khác (cùng aisle)
Bước 3: Lặp: từ Y, lại đi lên Aisle của Y → chọn Product Z khác
```

**Ví dụ:**
```
Sữa_tươi → Aisle_Sữa → Sữa_chua → Aisle_Sữa → Sữa_đặc → ...
```

**Tác dụng:** Học được rằng Sữa_tươi và Sữa_chua thuộc cùng gian hàng → embedding gần nhau, dù chúng có thể ít khi mua chung.

### 5.4. Code chi tiết

```python
for walk_idx in range(total_walks):
    # Quyết định kịch bản
    use_behavioral = np.random.random() < behavioral_ratio
    
    if use_behavioral:
        # --- Kịch bản 1: Behavioral (P → P → P) ---
        start_node = nodes_with_edges[walk_idx % n_behavioral]
    else:
        # --- Kịch bản 2: Semantic (P → A → P) ---
        start_node = nodes_with_aisle[walk_idx % n_semantic]
    
    cur = start_node
    walks_arr[walk_idx, 0] = cur
    
    for step in range(1, walk_length):
        if use_behavioral:
            # Behavioral: đi theo CO_OCCUR edge
            s, e = indptr[cur], indptr[cur + 1]
            n_nb = e - s
            if n_nb == 0:
                break  # hết đường
            nb_idx = rand_matrix[walk_idx, step] % n_nb
            cur = neighbors[s + nb_idx]
        else:
            # Semantic: P → A → P
            aisle_id = product_to_aisle.get(cur)
            if aisle_id is None:
                break  # fallback
            products_in_aisle = aisle_to_products[aisle_id]
            candidates = [p for p in products_in_aisle if p != cur]
            if candidates:
                cur = candidates[rand_matrix[walk_idx, step] % len(candidates)]
            else:
                break
        
        walks_arr[walk_idx, step] = cur
```

### 5.5. Kết quả sau walk

Sau khi walk xong, ta có mảng walks:
```
walks = [
    [101, 102, 103, 104, 105, ...],  # Walk 1: behavioral
    [101, 201, 203, 204, 205, ...],  # Walk 2: semantic
    [102, 103, 104, 101, 105, ...],  # Walk 3: behavioral
    ...
]
```

Mỗi walk là 1 "câu" — độ dài = walk_length (hoặc ngắn hơn nếu hết đường).

### 5.6. Xử lý fallback

Khi đi walk có thể gặp:
- **Behavioral:** node không có neighbor CO_OCCUR → reset về start hoặc chuyển semantic
- **Semantic:** node không có aisle → fallback về behavioral
- **Semantic:** aisle chỉ có 1 sản phẩm (chính nó) → fallback về behavioral

---

## 6. Bước 4: Word2Vec (Skip-gram + Negative Sampling)

### 6.1. Đây chính là Item2Vec

Sau khi có các walks (dạng sequences), ta đưa vào **Word2Vec (Skip-gram)** — hoàn toàn giống Item2Vec.

Khác biệt duy nhất: "câu" không phải đơn hàng thật, mà là metapath walks.

### 6.2. Chuẩn bị sentences

```python
sentences = []
for i in range(walks.shape[0]):
    wlen = walk_lengths[i]
    if wlen > 1:
        sentence = [str(walks[i, j]) for j in range(wlen)]
        sentences.append(sentence)
```

### 6.3. Train Word2Vec

```python
self.model = Word2Vec(
    sentences=sentences,
    vector_size=embedding_dim,  # 128
    window=window,              # 5
    min_count=1,                # giữ tất cả product nodes
    negative=negative,          # 10
    epochs=epochs,              # 10
    workers=workers,            # 4
    sg=1,                       # Skip-gram
)
```

### 6.4. Chi tiết cách Item2Vec hoạt động

> **Quay lại file `docs/item2vec.md` để xem giải thích chi tiết từ A-Z về:**
> - Skip-gram là gì
> - Window và context
> - Khởi tạo embedding ngẫu nhiên
> - Forward pass: tính Loss
> - Backward pass: tính Gradient
> - Cập nhật embedding
> - Negative Sampling
> - Cosine Similarity
> - Ví dụ bằng số đầy đủ

---

## 7. Bước 5: Recommend bằng Cosine Similarity

### 7.1. Lấy embedding

Sau khi train Word2Vec, mỗi sản phẩm có 1 embedding vector 128 chiều:

```python
self.embeddings = np.zeros((n_products, embedding_dim))
for node in range(n_products):
    pid = self.idx_to_product_id[node]
    pid_str = str(pid)
    if pid_str in self.model.wv:
        self.embeddings[node] = self.model.wv[pid_str]
```

### 7.2. Tính Cosine Similarity

```python
similarities = (self.embeddings @ vec_a) / (self._embedding_norms * norm_a)
```

Trong đó:
- `self.embeddings @ vec_a` = dot product giữa tất cả embeddings với vector đầu vào
- `self._embedding_norms` = độ dài (norm) của mỗi embedding (đã cache)
- `norm_a` = độ dài của vector đầu vào

### 7.3. Recommend

```python
similarities[idx] = -1  # bỏ qua chính nó
top_indices = np.argsort(similarities)[::-1][:top_k]
```

---

## 8. Ví dụ chi tiết bằng số

### 8.1. Dữ liệu

```
5 sản phẩm: Sữa(0), Bánh_mì(1), Trứng(2), Bơ(3), Mứt(4)
2 Aisle: Sữa(A1) chứa {Sữa, Sữa_chua}, Bánh(A2) chứa {Bánh_mì, Bơ}
```

### 8.2. Co-occurrence counts (sau threshold)

```
Sữa ↔ Bánh_mì: 100 lần
Sữa ↔ Trứng: 80 lần
Bánh_mì ↔ Bơ: 60 lần
Trứng ↔ Bơ: 40 lần
```

### 8.3. Adjacency list

```
Sữa:     [(Bánh_mì, 100), (Trứng, 80)]
Bánh_mì: [(Sữa, 100), (Bơ, 60)]
Trứng:   [(Sữa, 80), (Bơ, 40)]
Bơ:      [(Bánh_mì, 60), (Trứng, 40)]
Mứt:     []
```

### 8.4. Metapath walks (behavioral ratio = 0.7)

**Walk 1 (behavioral):** bắt đầu từ Sữa
```
Sữa → Bánh_mì → Bơ → Trứng → Sữa → ...
```

**Walk 2 (semantic):** bắt đầu từ Sữa
```
Sữa → Aisle_Sữa → Sữa_chua → Aisle_Sữa → Sữa → ...
```

### 8.5. Sentences cho Word2Vec

```python
sentences = [
    [0, 1, 3, 2, 0, ...],  # Walk 1
    [0, 4, 2, 1, ...],      # Walk 2 (Sữa → semantic → Sữa_chua → ...)
    [1, 0, 2, 3, ...],      # Walk 3
    ...
]
```

### 8.6. Word2Vec (Item2Vec) training

> Chi tiết từ đây: xem **`docs/item2vec.md`**

Embedding học được sẽ phản ánh:
- Sữa và Bánh_mì gần nhau (hay đi cùng trong behavioral walks)
- Sữa và Sữa_chua gần nhau (hay đi cùng trong semantic walks)
- Mứt có thể gần các sản phẩm khác hơn (dù không có co-occurrence trực tiếp)

---

## 9. KG Metapath vs Item-CF (Ochiai) vs Item2Vec thuần

### 9.1. Bảng so sánh

| Tiêu chí | Item-CF (Ochiai) | Item2Vec thuần | KG Metapath |
|----------|-----------------|---------------|-------------|
| **Dữ liệu** | Co-occurrence counts | Đơn hàng thật | Đơn hàng + Aisle + Dept |
| **Cách học** | Công thức toán | Skip-gram trên đơn hàng | Skip-gram trên KG walks |
| **Quan hệ trực tiếp** | ✅ | ✅ | ✅ |
| **Quan hệ gián tiếp** | ❌ | ✅ (qua cầu nối) | ✅ (qua KG) |
| **Quan hệ cùng loại** | ❌ | ❌ | ✅ (qua semantic walk) |
| **Cold-start** | ❌ | ❌ | ❌ |
| **Tốc độ train** | ⚡ Nhanh | 🐢 Trung bình | 🐢🐢 Chậm nhất |

### 9.2. Ví dụ so sánh

Giả sử: Sữa và Sữa_chua thuộc cùng Aisle "Sữa tươi", nhưng **ít khi mua chung** (vì 1 lần chỉ mua 1 loại sữa).

| Model | similarity(Sữa, Sữa_chua) |
|-------|--------------------------|
| **Item-CF (Ochiai)** | Thấp (ít co-occurrence) |
| **Item2Vec thuần** | Thấp (ít xuất hiện cùng đơn) |
| **KG Metapath** | **Cao** (semantic walk: Sữa→Aisle_Sữa→Sữa_chua) |

Đây là điểm mạnh nhất của KG Metapath: học được quan hệ "cùng loại" dù không mua chung.

### 9.3. Khi nào dùng cái gì?

| Tình huống | Model phù hợp |
|------------|--------------|
| Cần real-time, đơn giản | Item-CF (Ochiai) |
| Có nhiều dữ liệu, muốn học quan hệ gián tiếp | Item2Vec thuần |
| Muốn tận dụng thông tin danh mục (aisle, department) | **KG Metapath** |
| Muốn kết hợp cả 3 | Ensemble (bước 6) |

---

## 10. Ưu điểm & Nhược điểm

### 10.1. Ưu điểm

1. **Tận dụng tri thức miền:** Không chỉ dùng co-occurrence, còn dùng thông tin aisle/department.

2. **Học quan hệ cùng loại:** Semantic walk giúp các sản phẩm cùng aisle gần nhau, dù ít mua chung.

3. **Kết hợp 2 kịch bản:** Behavioral (hành vi mua hàng) + Semantic (cấu trúc danh mục) → embedding giàu thông tin.

4. **Có thể mở rộng:** Dễ thêm loại nút mới (brand, category con...) và kịch bản walk mới.

### 10.2. Nhược điểm

1. **Chậm nhất pipeline:** Phải xây graph + sinh walks + train Word2Vec — mất nhiều thời gian nhất.

2. **Hyperparameters phức tạp:** Nhiều tham số cần tuning (walk_length, num_walks, behavioral_ratio, edge_threshold, window, negative, epochs...).

3. **Phụ thuộc vào chất lượng KG:** Nếu mapping product→aisle không chính xác, semantic walk sẽ học sai.

4. **Không interpretable:** Embedding là black box, khó giải thích tại sao 2 sản phẩm giống nhau.

### 10.3. Các tham số quan trọng

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|-----------------|---------|
| `embedding_dim` | 128 | Số chiều embedding |
| `walk_length` | 20 | Độ dài mỗi metapath walk |
| `num_walks` | 5 | Số walks từ mỗi node |
| `edge_threshold` | 3 | Lọc cạnh CO_OCCUR yếu |
| `window` | 5 | Context window cho Word2Vec |
| `negative` | 10 | Số negative samples |
| `epochs` | 10 | Số epoch Word2Vec |
| `behavioral_ratio` | 0.7 | Tỉ lệ behavioral vs semantic walks |

---

## Liên kết

- [Item2Vec chi tiết (cách Word2Vec hoạt động)](docs/item2vec.md)
- [Item-CF (Ochiai) chi tiết](docs/models.md#bước-3-item-based-collaborative-filtering-item-cf)
- [Sơ đồ KG](docs/kg_diagram.html)