# Mục Đích Dự Án: Hệ Thống Gợi Ý Mua Kèm (Bundle Recommendation)

## 1. Giới Thiệu Dự Án

Dự án này nhằm xây dựng một **Hệ thống Gợi ý Mua kèm (Bundle Recommendation System)** dựa trên tập dữ liệu Instacart Market Basket Analysis với hơn **33 triệu giao dịch** từ hơn **206,000 người dùng** và **49,000 sản phẩm**.

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

### 3.1 Toàn bộ dữ liệu dùng để TRAIN

| File | Số dòng | Vai trò |
|------|---------|---------|
| `order_products__prior.csv` | **32,434,490** | Dữ liệu lịch sử chính → học mối quan hệ sản phẩm |
| `order_products__train.csv` | **1,384,618** | Bổ sung thêm dữ liệu train |
| **Tổng cộng** | **33,819,108** | **Toàn bộ dùng để train model** |

### 3.2 Không dùng eval_set=test để đánh giá

- `eval_set=test` (~75,000 đơn hàng) sẽ **KHÔNG** được dùng để đánh giá model gợi ý mua kèm
- Lý do: Đây là dữ liệu kiểu sequential, không phải mua kèm

### 3.3 Đánh giá bằng khảo sát thực tế (Human Evaluation)

Thay vào đó, chất lượng gợi ý mua kèm sẽ được đánh giá thông qua **khảo sát người dùng thực tế**. Quy trình khảo sát chi tiết (phương pháp thu thập, cấu trúc dữ liệu, chỉ số đánh giá) được trình bày ở **Mục 5**.


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
                       │ Dùng: TF-IDF multi-field     │
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

#### Cải tiến: Ensemble Count Vectorizer (Overlap Score) + TF-IDF

CB hiện tại sử dụng **ensemble của Count Vectorizer (Overlap Score) và TF-IDF (Cosine similarity)**:

```python
final_similarity = alpha × Overlap(Count) + (1 - alpha) × Cosine(TF-IDF)
```

| Thành phần | Vai trò |
|---|---|
| **Count Vectorizer** (raw count → Overlap) | Đếm từ đơn tuyệt đối: càng trùng nhiều từ → điểm càng tiến sát 1.0. Bất chấp độ dài câu hay trật tự từ |
| **TF-IDF** (Cosine) | Phân hóa thông minh: từ khóa chính (Cá hồi, Bắp bò) ↑, từ phụ (gói, hộp) ↓ |
| **Overlap metric** (Overlap Score) | `|A ∩ B| / min(|A|, |B|)` trên nhị phân hóa — chỉ quan tâm tỷ lệ từ trùng so với bên ít từ hơn |
| **Alpha** (`CB_ALPHA = 0.5`) | Cân bằng Overlap Count + TF-IDF (mỗi bên 50%) |

Cả 2 ma trận đều là sparse CSR, lưu riêng thành 2 file:
- `tfidf_vectors.npz` — ma trận TF-IDF (đã L2-normalized)
- `count_vectors.npz` — ma trận Count raw (chưa normalize, dùng cho Overlap Score)

Tại inference, similarity được tính on-demand bằng công thức ensemble, không pre-compute full matrix.

---

#### Vấn đề cần giải quyết

Các model co-occurrence học từ hành vi mua hàng thực tế. Khi người dùng thường mua nhiều loại bia cùng lúc, các model này sẽ học được "Bia A → Bia B". Nhưng đây là **substitute** (sản phẩm thay thế), không phải **complementary** (mua kèm thực sự). CB phát hiện và loại bỏ những trường hợp như vậy.

---

#### Cách hoạt động

**Bước 1 — Inference-time (không pre-compute full matrix)**

Thay vì tính trước toàn bộ ma trận similarity 49K × 49K (tốn ~10GB RAM), CB chỉ tính similarity **on-demand** cho các cặp (A, B) mà co-occurrence models thực sự đề xuất:

```python
# Vector hóa từng sản phẩm (pre-compute 1 lần)
# TF-IDF trên product_name (unigram + bigram)
product_vectors = vectorize_all_products(products_df)  # shape: (49K, D)

# Tại inference: chỉ tính similarity cho top candidates
def cb_similarity(product_a_id, candidate_ids):
    vec_a = product_vectors[product_a_id]
    vecs_b = product_vectors[candidate_ids]
    return cosine_similarity(vec_a, vecs_b)  # chỉ len(candidates) phép tính
```

**Bước 2 — Lọc substitute (khi dùng hybrid ensemble)**

CB tự nó không biết ngưỡng — đây là việc của ensemble:
- Tính `CB_similarity(A, B)` = cosine similarity của vector mô tả
- Nếu `similarity = 0` → hoàn toàn khác nhau, không có thông tin → **loại bỏ**
- Nếu `0 < similarity < threshold` → B là complementary → **giữ lại**
- Nếu `similarity ≥ threshold` → B là substitute → **loại bỏ**

**Ví dụ thực tế**:

| Cặp sản phẩm | CB Similarity | Kết quả |
|---|---|---|
| Budweiser + Corona | 0.91 (cùng aisle beers, tên gần) | ❌ Loại (substitute) |
| Budweiser + Doritos | 0.12 (khác aisle, tên khác xa) | ✅ Giữ (complementary) |
| Whole Milk + 2% Milk | 0.88 (cùng aisle dairy) | ❌ Loại (substitute) |
| Whole Milk + Cereal | 0.09 (khác department) | ✅ Giữ (complementary) |
| Beer + Carrot | 0.00 (hoàn toàn khác) | ❌ Loại (không thông tin) |

---

#### Ưu điểm
- **Tiết kiệm bộ nhớ**: Chỉ tính similarity khi cần, không lưu full matrix
- **Loại bỏ substitute hiệu quả**: Dùng tên + aisle + department, phát hiện substitute ngay cả khi chưa có dữ liệu mua hàng
- **Cold-start & Long-tail**: Sản phẩm mới/ít giao dịch vẫn được lọc đúng nhờ thông tin mô tả

#### Nhược điểm
- Chỉ dựa trên thông tin mô tả, không capture hành vi mua hàng thực tế
- Similarity = 0 (tên hoàn toàn khác) không có thông tin → phải loại bỏ thủ công

> **Lưu ý về threshold**: CB **không có threshold riêng**. Ngưỡng lọc substitute là tham số của **ensemble** (hybrid), đặt tại `ENS_CB_THRESHOLD` trong config. Tác động của CB được đánh giá gián tiếp qua so sánh `Ensemble` vs `Ensemble + CB Filter`.

---

### 4.2 Item-Based Collaborative Filtering (Item-CF) — Ochiai + Confidence Score

> **Ghi chú**: Ochiai coefficient = `cnt / sqrt(count(A) * count(B))` tương đương **Cosine similarity** trên binary vector (mỗi sản phẩm là binary vector với chiều là các order). Đây chính là **Item-Based Collaborative Filtering (Item-CF)**. Tuy nhiên, score tổng hợp thêm confidence (bất đối xứng, có hướng) và log-frequency (popularity bonus) để phù hợp hơn với bài toán gợi ý mua kèm có hướng.

**Ý tưởng**: Đếm số lần cặp sản phẩm (A, B) cùng xuất hiện trong một đơn hàng, sau đó tính score phản ánh cả **mức độ liên kết** lẫn **hướng gợi ý**.

---

#### Công thức

```python
# Ký hiệu
# cnt       = số đơn hàng chứa cả A và B
# count(A)  = số đơn hàng chứa A
# count(B)  = số đơn hàng chứa B

# 1. Ochiai Coefficient (Cosine Similarity trên binary co-occurrence)
#    Đối xứng: ochiai(A,B) = ochiai(B,A)
#    = Cosine similarity giữa 2 binary vector → Item-Based CF
#    Normalize theo cả hai item → không phạt item phổ biến như PMI
ochiai = cnt / sqrt(count(A) * count(B))

# 2. Log Frequency (Popularity Bonus)
#    Đối xứng: reward pair có volume lớn, tránh overfit pair hiếm
log_ab = log1p(cnt)   # = log(1 + cnt)

# 3. Confidence (Bất đối xứng → đồ thị có hướng)
#    Xác suất mua B khi đã chọn A (và ngược lại)
conf_a_b = cnt / count(A)
conf_b_a = cnt / count(B)

# 4. Score cuối (có hướng)
score(A → B) = ochiai * conf_a_b * log_ab
score(B → A) = ochiai * conf_b_a * log_ab
```

> **Lưu ý**: `score(A → B) ≠ score(B → A)` vì confidence có hướng.
> Ví dụ: "Mua nước lẩu → gợi ý lẩu" mạnh hơn "Mua lẩu → gợi ý nước lẩu"
> vì gần như ai mua nước lẩu cũng mua lẩu (conf cao), nhưng không phải ai mua lẩu cũng mua nước lẩu đó.

---

#### Tại sao không dùng PMI / SPPMI?

| Vấn đề | PMI/SPPMI | Item-CF (Ochiai + Conf) |
|---|---|---|
| Item phổ biến bị underrate (cơm + gà) | ❌ PMI phạt nặng P(A) lớn | ✅ Ochiai normalize cân bằng (Item-Based CF) |
| Pair hiếm 1–2 lần bị overrate | ⚠️ SPPMI giảm nhưng chưa đủ | ✅ log1p suppress + min_support |
| Chỉ cho score đối xứng | ✅/❌ | ✅ Confidence có hướng |
| Substitute detection qua PMI âm | ✅ | ❌ Nhưng CB filter đã lo phần này |

---

#### Min-support filter

Trước khi tính score, loại bỏ các pair có count quá thấp:

```python
MIN_SUPPORT = 10  # pair phải xuất hiện ít nhất 10 lần
if cnt < MIN_SUPPORT:
    continue  # bỏ qua pair này
```

Lý do: pair xuất hiện < 10 lần trên 33M giao dịch là noise thống kê, không đáng tin.

---

#### Xử lý kỹ thuật

- **Dữ liệu**: 33M records, 49K products → dùng **CSR sparse matrix** để lưu co-occurrence counts
- **Gợi ý**: Top-K sản phẩm có `score(A → B)` cao nhất với sản phẩm đầu vào A

#### Ưu điểm
- Dễ implement, tính toán nhanh
- Score có hướng phản ánh đúng thực tế gợi ý
- Không bị bias bởi item phổ biến như PMI

#### Nhược điểm
- Chỉ capture mối quan hệ cặp đôi trực tiếp
- Không capture được mối quan hệ gián tiếp (A→B→C)
- Không capture được ngữ nghĩa sản phẩm

---

### 4.3 Item2Vec (Neural Item-Based CF) — Word2Vec Skip-gram trên Giỏ Hàng

**Ý tưởng**: Coi mỗi đơn hàng (order) như một **"câu"**, mỗi sản phẩm (product) là một **"từ"**. Áp dụng Word2Vec để học embedding cho từng sản phẩm từ ngữ cảnh mua hàng.

---

#### Cách hoạt động

```
Đơn hàng #1: [Sữa, Bánh mì, Bơ, Trứng]
Đơn hàng #2: [Bia, Snack, Pizza]
Đơn hàng #3: [Sữa, Ngũ cốc, Chuối]
     ↓
Word2Vec Skip-gram học:
- "Sữa" hay xuất hiện gần "Bánh mì", "Bơ", "Ngũ cốc"
  → embedding của "Sữa" gần với những sản phẩm đó
- "Bia" hay xuất hiện gần "Snack", "Pizza"
  → embedding của "Bia" gần với "Snack", "Pizza"
```

**Model**: Skip-gram với Negative Sampling — thư viện `gensim`

**Hyperparameter quan trọng**:

| Param | Ý nghĩa | Gợi ý |
|---|---|---|
| `vector_size` | Số chiều embedding | 100–300 |
| `window` | Số sản phẩm lân cận trong order | 5–10 |
| `min_count` | Sản phẩm xuất hiện ít hơn sẽ bị bỏ qua | 5–20 |
| `negative` | Số negative samples | 5–20 |
| `epochs` | Số lần lặp | 5–30 |

**Gợi ý**: Top-K sản phẩm có cosine similarity cao nhất với embedding của sản phẩm đầu vào.

---

#### Ưu điểm
- Học được **ngữ nghĩa sản phẩm từ ngữ cảnh mua hàng** — không chỉ đếm cặp trực tiếp
- Capture được **mối quan hệ gián tiếp**: A hay đi với B, B hay đi với C → A có liên quan đến C
- Embedding có thể tái sử dụng cho nhiều downstream tasks

#### Nhược điểm
- Cần tuning nhiều hyperparameter
- Item2Vec không phân biệt được hướng (đối xứng, không như Item-CF có hướng)
- Sản phẩm có `count < min_count` sẽ không có embedding (long-tail issue)

---

### 4.4 Graph-based: KGMetapath (KG Embedding + Metapath Walk)

**Ý tưởng**: Xây dựng **Đồ thị Tri thức đa thể (Heterogeneous Knowledge Graph - KG)** với 3 loại nút (Product, Aisle, Department) và 3 loại quan hệ (CO_OCCUR, BELONGS_TO, PART_OF). Sử dụng **Metapath Walk** — duyệt đồ thị định hướng ngữ nghĩa — thay vì uniform random walk để giải quyết bài toán siêu nút (supernode) và long-tail.

---

#### Cấu trúc KG

```
Product (P) — 49,000 nodes
Aisle (A)   — 134 nodes  
Department (D) — 21 nodes

Cạnh:
- P1 --CO_OCCUR--> P2  (đồng xuất hiện ≥ 10 lần, từ ma trận thưa)
- P  --BELONGS_TO--> A   (quan hệ danh mục)
- A  --PART_OF--> D      (quan hệ phân cấp)
```

---

#### Kịch bản Metapath Walk

**Kịch bản 1 — Behavioral (học hành vi đồng xuất hiện)**:
```
P1 --CO_OCCUR--> P2 --CO_OCCUR--> P3
```
Robot đi theo các cạnh CO_OCCUR, bắt trọn vẹn các mẫu kết hợp giỏ hàng thực tế.

**Kịch bản 2 — Semantic (cứu vớt hàng hiếm đuôi dài)**:
```
P1 --BELONGS_TO--> A --random--> P2
```
Robot thoát khỏi vòng lặp sản phẩm phổ biến thông qua nút Aisle, tiếp cận các sản phẩm hiếm (46.3% sản phẩm có dưới 50 lần tương tác).

Tỉ lệ mẫu: 50% Behavioral, 50% Semantic (tham số `MW_METAPATH_BEHAVIORAL_RATIO`).

---

#### Cách hoạt động

**Bước 1 — Xây KG**:
- Đồ thị CO_OCCUR: từ co-occurrence count
- Mapping Product → Aisle: từ dữ liệu gốc
- Mapping Aisle → Products: index ngược cho semantic walk

**Bước 2 — Metapath Walk**:
Mỗi walk bắt đầu từ một product node, quyết định kịch bản theo tỉ lệ cấu hình:
- Behavioral: uniform random trên neighbors CO_OCCUR
- Semantic: P → A → P (chọn sản phẩm khác trong cùng aisle)

**Bước 3 — Học embedding**:
Áp dụng Word2Vec (Skip-gram + Negative Sampling) lên các chuỗi Metapath Walk → embedding 128 chiều cho mỗi sản phẩm. Tầng Walk là white-box tự viết, tầng Skip-gram dùng thư viện Gensim.

---

#### Ưu điểm
- **Giải quyết triệt để supernode**: Không bị kẹt ở Aisle/Department nhờ lộ trình cố định
- **Xử lý long-tail**: Sản phẩm hiếm vẫn được kết nối không gian qua quan hệ danh mục
- **High-order relationships**: Giữ được ưu điểm của graph embedding
- **White-box walk**: Kiểm soát minh bạch từng bước toán học

#### Nhược điểm
- Phức tạp hơn DeepWalk trong việc xây dựng đồ thị đa thể
- Cần tuning thêm tham số `behavioral_ratio`

---

### 4.5 Hybrid Ensemble (Kết hợp Co-occurrence Models)

**Ý tưởng**: Mỗi model co-occurrence capture một khía cạnh khác nhau của mối quan hệ sản phẩm. Kết hợp chúng bằng weighted score để tận dụng điểm mạnh của từng model.

---

#### Công thức ensemble

```
final_score(A → B) = α × ItemCF_score(A, B)
                   + β × Item2Vec_sim(A, B)
                   + γ × Metapath2Vec_sim(A, B)
```

Trong đó:
- `α, β, γ` là các trọng số, `α + β + γ = 1`
- Mỗi score được **normalize về [0, 1]** trước khi kộp để tránh một model dominate

**Đóng góp của từng model**:

| Model | Khía cạnh đóng góp |
|---|---|
| **Item-CF (Ochiai + Conf)** | Mối quan hệ đồng xuất hiện trực tiếp, có hướng, robust với item phổ biến |
| **Item2Vec** | Ngữ nghĩa sản phẩm từ context mua hàng, mối quan hệ gián tiếp |
| **KGMetapath** | Mối quan hệ bậc cao trong cấu trúc đồ thị KG, xử lý long-tail nhờ semantic walk |

> **CB (TF-IDF) không tham gia ensemble** — CB là tầng hậu xử lý độc lập, áp dụng sau khi có kết quả ensemble.

---

#### Normalize trước khi ensemble

Vì các model có range score khác nhau, cần normalize về [0, 1]:

```python
# Min-max normalization cho mỗi model
def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    return [(s - min_s) / (max_s - min_s + 1e-9) for s in scores]
```

---

### 4.6 Tiêu Chí So Sánh

| Tiêu chí | Item-CF | Item2Vec | KGMetapath | CB Filter | Ensemble (w/o CB) | Ensemble + CB |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Loại CF** | Memory-based Item-CF ✅ | Neural Item-Based CF ✅ | Không phải CF (Graph-based) ❌ | — | — | — |
| **Vai trò** | Model gợi ý | Model gợi ý | Model gợi ý | Bộ lọc hậu xử lý | Ensemble thuần (ICF+I2V+MW) | Ensemble + Filter |
| **LLM Eval (Precision@10)** | ? | ? | ? | — ⚠️ | ? | ? (kỳ vọng cao nhất) |
| **Loại bỏ substitute** | ❌ | ❌ | ❌ | ✅ Tốt | ❌ | ✅ Tốt |
| **Item phổ biến (cơm+gà)** | ✅ Item-CF fix | ✅ | ✅ | — | ✅ | ✅ |
| **Pair hiếm / noise** | ✅ min_support | ⚠️ min_count | ⚠️ threshold | — | ✅ | ✅ |
| **Cold-start** | ❌ | ❌ | ⚠️ (qua aisle) | ✅ (filter) | ❌ | ✅ |
| **Long-tail** | ⚠️ | ⚠️ | ✅ (semantic walk) | ✅ (filter) | ⚠️ | ✅ |
| **Hướng gợi ý (A→B ≠ B→A)** | ✅ | ❌ | ❌ | — | ✅ (từ Item-CF) | ✅ (từ Item-CF) |
| **Khả năng giải thích** | Cao | TB | Thấp | Cao | TB | TB |
| **Thời gian train** | Nhanh | TB | Chậm | Nhanh | TB | TB |

> ⚠️ **CB Filter không có LLM Eval riêng**: CB là deterministic filter, tác động của nó được đánh giá **gián tiếp** qua việc so sánh `Ensemble (w/o CB)` vs `Ensemble + CB`. Nếu Ensemble+CB có Precision@10 cao hơn, chứng tỏ CB đang loại bỏ substitute hiệu quả.
---

### 4.7 Thứ Tự Triển Khai

| Bước | Việc cần làm | Lý do |
|---|---|---|
| **1** | **CB — vector hóa sản phẩm** | Pre-compute product vectors (TF-IDF + one-hot). Cần xong trước để dùng làm filter ở bước 7 |
| **2** | **Item-CF (Item-Based CF)** | Nhanh, hiệu quả, là backbone của ensemble |
| **3** | **Item2Vec** | Bổ sung ngữ nghĩa cho ensemble |
| **4** | **KGMetapath** | Xây KG + Metapath Walk, bổ sung high-order relationships + xử lý long-tail |
| **5** | **Co-occurrence Ensemble** | Kết hợp bước 2+3+4 bằng weighted score |
| **6** | **CB Diversity Filter** | Áp dụng CB filter lên kết quả ensemble, loại substitute |

Mỗi bước đều có thể **chạy độc lập** và **đánh giá riêng** qua LLM evaluation (xem **Mục 5**) trước khi tiến sang bước tiếp theo.

---

## 5. Survey Dataset — Ground Truth cho Human Evaluation bằng LLM

### 5.1 Tổng quan

Sau khi train xong tất cả các model, cần một tập **ground truth** để đánh giá định lượng chất lượng gợi ý mua kèm. Do bài toán gợi ý mua kèm là **phi tuần tự** và không có sẵn nhãn từ dữ liệu Instacart gốc, tập ground truth này được xây dựng thông qua khảo sát sử dụng **Model Ngôn Ngữ Lớn (LLM)** làm người đánh giá.

LLM đóng vai trò như một "người dùng thông thái" — dựa trên kiến thức về sản phẩm và thói quen mua sắm, LLM đưa ra nhận xét liệu một cặp sản phẩm có thực sự **mua kèm (complementary)** hay không.

---

### 5.2 Phương pháp thu thập

#### 5.2.1 Cấu trúc mẫu khảo sát

Mỗi mẫu khảo sát là một cặp `(product_A, product_B)`. LLM được hỏi: *"Nếu khách hàng mua product_A, liệu họ có khả năng mua product_B trong cùng giỏ hàng không?"* — trả lời **Có (1)** hoặc **Không (0)**.

#### 5.2.2 Cách chọn mẫu

Với mỗi model cần đánh giá (Item-CF, Item2Vec, KGMetapath, Ensemble+CB), mẫu được chọn theo tỷ lệ:

| Loại mẫu | Tỷ lệ | Cách chọn |
|---|---|---|
| **Top-5** | 50% | Chọn `product_A` ngẫu nhiên, lấy `product_B` từ top-5 gợi ý của model |
| **Nhiễu (Noise)** | 50% | Chọn `product_A` ngẫu nhiên, lấy `product_B` ngẫu nhiên từ các sản phẩm **không nằm** trong top-K gợi ý của model |

Lý do tỷ lệ 50-50:
- **Top-5**: Đo khả năng model gợi ý đúng sản phẩm mua kèm
- **Nhiễu**: Đo khả năng model **không** gợi ý sản phẩm không liên quan (kiểm tra precision âm)

Tất cả các model được đánh giá trên **cùng một bộ mẫu** (cùng seed ngẫu nhiên) để đảm bảo công bằng.

#### 5.2.3 LLM làm người đánh giá

- Sử dụng một LLM có kiến thức đa dạng về sản phẩm tiêu dùng (ví dụ: GPT-4, Claude, Gemini)
- LLM được cung cấp tên sản phẩm (`product_name`), aisle, department để có đủ ngữ cảnh
- Mỗi cặp được hỏi độc lập, không tiết lộ model nào tạo ra mẫu đó
- Có thể yêu cầu LLM giải thích ngắn lý do (optional) để kiểm tra chất lượng đánh giá

---

### 5.3 Cấu trúc dữ liệu (Schema)

Dữ liệu khảo sát được lưu trong thư mục `data/survey/`, mỗi model một file CSV riêng (hoặc chung với cột `model_name`). Schema dự kiến:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `product_A_id` | int | ID sản phẩm đầu vào |
| `product_B_id` | int | ID sản phẩm được hỏi |
| `model_name` | str | Tên model tạo ra mẫu này (`item_cf`, `item2vec`, `kg_metapath`, `ensemble_cb`) |
| `source` | str | `"top5"` hoặc `"noise"` |
| `llm_label` | int | 1 = complementary, 0 = not complementary |
| `llm_reasoning` | str (optional) | Giải thích ngắn từ LLM về lý do đánh giá |

---

### 5.4 Các chỉ số đánh giá (Metrics)

Tất cả các chỉ số được tính trên top-10 gợi ý của mỗi model, sử dụng ground truth từ LLM.

#### 5.4.1 Precision@10

```python
Precision@10 = số lượng gợi ý đúng (complementary) trong top-10 / 10
```

**Ý nghĩa**: Trong số 10 sản phẩm được gợi ý, có bao nhiêu sản phẩm thực sự là mua kèm?

#### 5.4.2 Recall@10

```python
Recall@10 = số lượng gợi ý đúng trong top-10 / tổng số sản phẩm complementary (trong ground truth)
```

**Ý nghĩa**: Model đã gợi ý được bao nhiêu phần trăm sản phẩm mua kèm thực sự?

#### 5.4.3 F1@10

```python
F1@10 = 2 × (Precision@10 × Recall@10) / (Precision@10 + Recall@10)
```

**Ý nghĩa**: Trung bình điều hòa giữa Precision và Recall — đánh giá tổng thể.

#### 5.4.4 Hit@10

```python
Hit@10 = 1 nếu có ít nhất 1 gợi ý đúng trong top-10, ngược lại = 0
```

**Ý nghĩa**: Model có gợi ý được ít nhất một sản phẩm mua kèm hữu ích hay không?

---

### 5.5 Cách dùng để so sánh model

Mỗi model được đánh giá trên cả 4 chỉ số. Kết quả được tổng hợp trong bảng so sánh:

| Model | Precision@10 | Recall@10 | F1@10 | Hit@10 |
|---|---|---|---|---|
| Item-CF (Item-Based CF) | ⬜ | ⬜ | ⬜ | ⬜ |
| Item2Vec | ⬜ | ⬜ | ⬜ | ⬜ |
| KGMetapath | ⬜ | ⬜ | ⬜ | ⬜ |
| Ensemble (w/o CB) | ⬜ | ⬜ | ⬜ | ⬜ |
| **Ensemble + CB Filter** | ⬜ | ⬜ | ⬜ | ⬜ |

> **Kỳ vọng**: Ensemble + CB Filter đạt điểm cao nhất ở tất cả các chỉ số nhờ kết hợp sức mạnh của nhiều model và bộ lọc substitute. So sánh `Ensemble (w/o CB)` vs `Ensemble + CB` cho thấy tác động của CB filter trong việc loại bỏ substitute và cải thiện chất lượng gợi ý.