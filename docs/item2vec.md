# Item2Vec — Giải Thích Chi Tiết Từ A-Z

> Tài liệu này giải thích thuật toán Item2Vec một cách cực kỳ chi tiết, từ khái niệm, cách hoạt động, công thức toán, ví dụ số cụ thể, đến code minh họa. Phù hợp để chuẩn bị trả lời câu hỏi bảo vệ đồ án.

---

## Mục lục

1. [Item2Vec là gì?](#1-item2vec-là-gì)
2. [Nguồn gốc: Word2Vec](#2-nguồn-gốc-word2vec)
3. [Cách hoạt động tổng quan](#3-cách-hoạt-động-tổng-quan)
4. [Bước 1: Chuẩn bị dữ liệu](#4-bước-1-chuẩn-bị-dữ-liệu)
5. [Bước 2: Tạo training pairs (Skip-gram)](#5-bước-2-tạo-training-pairs-skip-gram)
6. [Bước 3: Khởi tạo Embedding](#6-bước-3-khởi-tạo-embedding)
7. [Bước 4: Negative Sampling](#7-bước-4-negative-sampling)
8. [Bước 5: Forward Pass — Tính Loss](#8-bước-5-forward-pass--tính-loss)
9. [Bước 6: Backward Pass — Tính Gradient](#9-bước-6-backward-pass--tính-gradient)
10. [Bước 7: Cập nhật Embedding](#10-bước-7-cập-nhật-embedding)
11. [Bước 8: Lặp lại qua nhiều Epoch](#11-bước-8-lặp-lại-qua-nhiều-epoch)
12. [Bước 9: Tính Similarity & Recommend](#12-bước-9-tính-similarity--recommend)
13. [Ví dụ đầy đủ bằng số](#13-ví-dụ-đầy-đủ-bằng-số)
14. [Code minh họa với Gensim](#14-code-minh-họa-với-gensim)
15. [Item2Vec vs Item-CF (Ochiai)](#15-item2vec-vs-item-cf-ochiai)
16. [Ưu điểm & Nhược điểm](#16-ưu-điểm--nhược-điểm)
17. [Các câu hỏi thường gặp](#17-các-câu-hỏi-thường-gặp)

---

## 1. Item2Vec là gì?

**Item2Vec** là thuật toán học **embedding vector** (vector số) cho mỗi sản phẩm dựa trên hành vi mua hàng của khách hàng. Nó được đề xuất bởi **Oren Barkan và Noam Koenigstein** trong paper "Item2Vec: Neural Item Embedding for Collaborative Filtering" (2016).

**Ý tưởng chính:** Áp dụng **Word2Vec (Skip-gram + Negative Sampling)** — vốn dùng cho văn bản — vào các sản phẩm trong đơn hàng.

| NLP (Word2Vec) | Recommendation (Item2Vec) |
|----------------|--------------------------|
| 1 câu văn | 1 đơn hàng (order) |
| 1 từ (word) | 1 sản phẩm (item) |
| Ngữ cảnh = các từ xung quanh | Ngữ cảnh = các sản phẩm cùng đơn |
| Từ xuất hiện cùng ngữ cảnh → nghĩa giống | Sản phẩm xuất hiện cùng đơn → liên quan nhau |

**Kết quả:** Mỗi sản phẩm được biểu diễn bằng 1 vector số (VD 128 chiều). Sản phẩm nào hay được mua chung thì vector của chúng gần nhau trong không gian embedding.

---

## 2. Nguồn gốc: Word2Vec

### 2.1. Word2Vec là gì?

Word2Vec là thuật toán do **Tomas Mikolov (Google)** công bố năm 2013. Mục tiêu: biến mỗi **từ** thành 1 **vector số** (embedding) sao cho:

- Từ có nghĩa giống nhau → vector gần nhau (vd: "vua" gần "hoàng đế")
- Từ có quan hệ ngữ pháp → vector có cùng hướng (vd: vector("vua") - vector("nam") + vector("nữ") ≈ vector("hoàng hậu"))

### 2.2. Distributional Hypothesis

> **"Một từ được đặc trưng bởi ngữ cảnh xung quanh nó"**

Trong câu: "Tôi **ăn** cơm với thịt"
- Từ "ăn" có ngữ cảnh = ["Tôi", "cơm", "với", "thịt"]
- Từ "uống" cũng xuất hiện trong ngữ cảnh tương tự: "Tôi **uống** nước với đá"
- → "ăn" và "uống" có vector gần nhau

### 2.3. Skip-gram Architecture

Có 2 kiến trúc:
- **CBOW (Continuous Bag of Words):** Dự đoán từ trung tâm từ ngữ cảnh
- **Skip-gram:** Dự đoán ngữ cảnh từ từ trung tâm

Item2Vec dùng **Skip-gram**:

```
Input: "ăn" → dự đoán: ["Tôi", "cơm", "với", "thịt"]
```

---

## 3. Cách hoạt động tổng quan

```
DỮ LIỆU: 31.9M records (order_id, product_id)
    │
    ▼
Bước 1: Gom sản phẩm theo đơn hàng
    │ Mỗi đơn hàng = 1 list product_id
    ▼
Bước 2: Tạo training pairs (Skip-gram)
    │ Với mỗi center, lấy context trong window
    ▼
Bước 3: Khởi tạo embedding ngẫu nhiên
    │ W (center) và W' (context) — ma trận N×D
    ▼
Bước 4: Với mỗi pair (center, context):
    │ - 1 positive sample
    │ - K negative samples (unigram distribution)
    ▼
Bước 5: Forward → tính Loss
    │ Loss = -log(σ(v_c·v_p)) - Σ log(σ(-v_c·v_nk))
    ▼
Bước 6: Backward → tính Gradient
    │ ∂L/∂v_c, ∂L/∂v_p, ∂L/∂v_nk
    ▼
Bước 7: Update embedding
    │ W -= η × gradient
    ▼
Lặp lại Bước 4-7 cho tất cả pairs × nhiều epoch
    │
    ▼
KẾT QUẢ: Mỗi sản phẩm có 1 embedding vector 128D
    │
    ▼
RECOMMEND: similarity = cosine(vector_A, vector_B)
```

---

## 4. Bước 1: Chuẩn bị dữ liệu

### 4.1. Dữ liệu gốc

```
order_products.parquet (31,919,315 records)
┌──────────┬────────────┐
│ order_id │ product_id │
├──────────┼────────────┤
│ 1        │ 101        │  ← Sữa
│ 1        │ 102        │  ← Bánh mì
│ 1        │ 103        │  ← Trứng
│ 2        │ 101        │  ← Sữa
│ 2        │ 102        │  ← Bánh mì
│ 2        │ 105        │  ← Mứt
│ 3        │ 103        │  ← Trứng
│ 3        │ 104        │  ← Bơ
│ ...      │ ...        │
└──────────┴────────────┘
```

### 4.2. Gom theo đơn hàng

Mỗi đơn hàng là 1 list các product_id (thứ tự có thể là thời gian thêm vào giỏ):

```
orders = [
    [101, 102, 103],          # Order 1: Sữa, Bánh mì, Trứng
    [101, 102, 105],          # Order 2: Sữa, Bánh mì, Mứt
    [103, 104, 101],          # Order 3: Trứng, Bơ, Sữa
    [102, 105, 101],          # Order 4: Bánh mì, Mứt, Sữa
]
```

### 4.3. Tại sao cần thứ tự?

Trong Word2Vec gốc, thứ tự từ rất quan trọng (ngữ pháp). Trong Item2Vec, thứ tự sản phẩm trong đơn hàng thường là **thời gian quét mã** hoặc **thứ tự thêm vào giỏ** — không có ý nghĩa ngữ pháp. Vì vậy, nhiều implementation coi đơn hàng như 1 **bag** (tập hợp không thứ tự) và dùng window = độ dài đơn hàng.

---

## 5. Bước 2: Tạo training pairs (Skip-gram)

### 5.1. Window là gì?

**Window** là số lượng sản phẩm tối đa về bên trái và bên phải của sản phẩm trung tâm được coi là "ngữ cảnh".

Ví dụ với **window = 2** và đơn hàng [A, B, C, D, E]:

```
Với center = C (vị trí 2):
  Trái: vị trí 0,1 → [A, B]  (trong window=2)
  Phải: vị trí 3,4 → [D, E]  (trong window=2)
  Context = [A, B, D, E]
```

### 5.2. Tạo pairs cho từng center

Với đơn hàng [Sữa(0), Bánh_mì(1), Trứng(2)] và window=2:

| Center | Vị trí | Context (trong window) | Pairs tạo ra |
|--------|--------|----------------------|--------------|
| Sữa | 0 | Bánh_mì(1), Trứng(2) | (Sữa, Bánh_mì), (Sữa, Trứng) |
| Bánh_mì | 1 | Sữa(0), Trứng(2) | (Bánh_mì, Sữa), (Bánh_mì, Trứng) |
| Trứng | 2 | Sữa(0), Bánh_mì(1) | (Trứng, Sữa), (Trứng, Bánh_mì) |

**Tổng cộng: 6 pairs từ 1 đơn hàng 3 sản phẩm.**

### 5.3. Dynamic Window

Trong thực tế (Gensim), window không cố định. Với mỗi center, window được chọn **ngẫu nhiên** từ 1 đến `window_max`:

```python
# Với window=5, mỗi lần lấy context:
actual_window = random.randint(1, 5)
# Lấy actual_window sản phẩm bên trái + actual_window sản phẩm bên phải
```

**Tác dụng:**
- Window nhỏ: ưu tiên quan hệ gần (chặt chẽ hơn)
- Window lớn: bắt được quan hệ xa
- Trung bình: cân bằng giữa local và global context

### 5.4. Tổng số pairs từ toàn bộ dữ liệu

Với 3.3 triệu đơn hàng, trung bình 10 sản phẩm/đơn, window=5:

```
Số pairs ≈ 3,300,000 × 10 × (trung bình 8 context/center)
         ≈ 264,000,000 pairs
```

---

## 6. Bước 3: Khởi tạo Embedding

### 6.1. Kích thước

- **N** = số sản phẩm (36,181 trong dự án)
- **D** = số chiều embedding (thường 100-300, phổ biến 128)

### 6.2. Hai ma trận embedding

Mỗi sản phẩm có **2 embedding vectors** riêng biệt:

| Ma trận | Kích thước | Vai trò |
|---------|-----------|---------|
| **W** | N × D | Embedding khi sản phẩm là **center** |
| **W'** | N × D | Embedding khi sản phẩm là **context** |

Tách riêng giúp model học tốt hơn (giống 2 góc nhìn khác nhau cho cùng 1 sản phẩm).

### 6.3. Khởi tạo giá trị

Khởi tạo ngẫu nhiên với giá trị nhỏ:

```python
# Cách 1: Uniform
W = np.random.uniform(-0.5/D, 0.5/D, size=(N, D))

# Cách 2: Normal distribution
W = np.random.randn(N, D) * 0.01
```

**Ví dụ với N=5 sản phẩm, D=3:**

```
W (center embeddings):
         dim0   dim1   dim2
Sữa     [0.10, -0.20, 0.30]    ← dòng 0
Bánh_mì [-0.10, 0.40, 0.20]    ← dòng 1
Trứng   [0.50, 0.10, -0.30]    ← dòng 2
Bơ      [-0.20, -0.10, 0.60]   ← dòng 3
Mứt     [0.30, -0.40, 0.10]    ← dòng 4

W' (context embeddings):
         dim0   dim1   dim2
Sữa     [0.20, 0.10, -0.10]    ← dòng 0
Bánh_mì [-0.30, 0.20, 0.40]    ← dòng 1
Trứng   [0.10, -0.50, 0.20]    ← dòng 2
Bơ      [0.40, 0.30, -0.20]    ← dòng 3
Mứt     [-0.10, 0.50, 0.30]    ← dòng 4
```

---

## 7. Bước 4: Negative Sampling

### 7.1. Vấn đề với Softmax

Nếu dùng softmax, mỗi lần forward phải tính:

```
P(context | center) = exp(v_c · v'_context) / Σ_{t=1}^{N} exp(v_c · v'_t)
```

Với N = 36,181 sản phẩm → tính exp 36,181 lần cho mỗi pair → cực kỳ chậm.

### 7.2. Giải pháp: Negative Sampling

Thay vì tính softmax cho **tất cả** sản phẩm, chỉ tính cho:
- **1 positive sample:** (center, context thật) → muốn score cao
- **K negative samples:** (center, sản phẩm ngẫu nhiên) → muốn score thấp

K thường = 5-20. Paper gốc Word2Vec khuyến nghị:
- Dataset nhỏ: K = 5-20
- Dataset lớn: K = 2-5

### 7.3. Unigram Distribution

Negative samples **không** được chọn uniform random. Dùng **unigram distribution**:

```
P(item) = count(item)^α / Σ count(t)^α
```

Trong đó:
- **count(item)** = số lần sản phẩm xuất hiện trong toàn bộ dữ liệu
- **α** = 0.75 (mũ 0.75)

**Tại sao mũ 0.75?**
- Giảm bớt sự áp đảo của sản phẩm cực kỳ phổ biến
- Nếu không có mũ 0.75: Sữa (500K) có xác suất gấp 10 lần Hạt_tiêu (50K)
- Có mũ 0.75: Sữa (500K^0.75=5,623) chỉ gấp ~5 lần Hạt_tiêu (50K^0.75=1,057)

**Ví dụ tính unigram:**

| Sản phẩm | count | count^0.75 | P = count^0.75 / Σ |
|----------|-------|-----------|-------------------|
| Sữa | 500,000 | 5,623 | 0.300 |
| Bánh_mì | 400,000 | 4,756 | 0.254 |
| Trứng | 300,000 | 3,833 | 0.205 |
| Bơ | 200,000 | 2,828 | 0.151 |
| Mứt | 100,000 | 1,682 | 0.090 |
| **Tổng** | **1,500,000** | **18,722** | **1.000** |

### 7.4. Quy trình chọn negative samples

```
Với pair (Sữa, Bánh_mì):
  1. Tính P(item) cho mọi sản phẩm (unigram)
  2. Random K=2 sản phẩm theo phân phối P
  3. Nếu chọn trúng Bánh_mì (context thật) → bỏ, chọn lại
  4. Nếu chọn trúng Sữa (center) → bỏ, chọn lại (tùy implementation)

Kết quả: chọn được Bơ và Mứt
```

### 7.5. Tổng hợp samples

```
Positive: (Sữa, Bánh_mì) — center=0, context=1
Negative 1: (Sữa, Bơ)    — center=0, neg=3
Negative 2: (Sữa, Mứt)   — center=0, neg=4
```

---

## 8. Bước 5: Forward Pass — Tính Loss

### 8.1. Lấy vector

```
v_c  = W[0]     = [0.10, -0.20, 0.30]    (center = Sữa)
v_p  = W'[1]    = [-0.30, 0.20, 0.40]    (positive context = Bánh_mì)
v_n1 = W'[3]    = [0.40, 0.30, -0.20]    (negative 1 = Bơ)
v_n2 = W'[4]    = [-0.10, 0.50, 0.30]    (negative 2 = Mứt)
```

### 8.2. Tính dot product

```
dot_pos = v_c · v_p
        = 0.10×(-0.30) + (-0.20)×0.20 + 0.30×0.40
        = -0.03 + (-0.04) + 0.12
        = 0.05

dot_neg1 = v_c · v_n1
         = 0.10×0.40 + (-0.20)×0.30 + 0.30×(-0.20)
         = 0.04 + (-0.06) + (-0.06)
         = -0.08

dot_neg2 = v_c · v_n2
         = 0.10×(-0.10) + (-0.20)×0.50 + 0.30×0.30
         = -0.01 + (-0.10) + 0.09
         = -0.02
```

### 8.3. Tính sigmoid

Công thức sigmoid:

```
σ(x) = 1 / (1 + e^(-x))
```

Tính:

```
σ(dot_pos)   = σ(0.05)  = 1 / (1 + e^(-0.05))  = 1 / (1 + 0.951) = 1 / 1.951 = 0.5125
σ(-dot_neg1) = σ(0.08)  = 1 / (1 + e^(-0.08))  = 1 / (1 + 0.923) = 1 / 1.923 = 0.5200
σ(-dot_neg2) = σ(0.02)  = 1 / (1 + e^(-0.02))  = 1 / (1 + 0.980) = 1 / 1.980 = 0.5050
```

### 8.4. Công thức Loss

```
L = -log(σ(v_c · v_p)) - Σ_{k=1}^{K} log(σ(-v_c · v_nk))
```

**Giải thích:**
- **Term 1 (positive):** `-log(σ(dot_pos))` — muốn σ(dot_pos) → 1 (dot_pos lớn dương) → loss nhỏ
- **Term 2 (negative):** `-log(σ(-dot_neg))` — muốn σ(-dot_neg) → 1 (dot_neg lớn âm) → loss nhỏ

### 8.5. Tính Loss cụ thể

```
L = -log(0.5125) - log(0.5200) - log(0.5050)
  = -(-0.668) - (-0.654) - (-0.683)
  = 0.668 + 0.654 + 0.683
  = 2.005
```

**Ý nghĩa:** Loss = 2.005. Mục tiêu: giảm loss này xuống càng gần 0 càng tốt.

### 8.6. Loss khi model lý tưởng

Nếu model học tốt:
- dot_pos rất lớn (VD 5.0) → σ(5.0) = 0.993 → -log(0.993) = 0.007
- dot_neg rất âm (VD -5.0) → σ(5.0) = 0.993 → -log(0.993) = 0.007
- Loss ≈ 0.007 + 0.007 + 0.007 = 0.021 (rất thấp)

---

## 9. Bước 6: Backward Pass — Tính Gradient

### 9.1. Đạo hàm của Loss

Từ loss function:

```
L = -log(σ(v_c·v_p)) - Σ log(σ(-v_c·v_nk))
```

Đạo hàm theo từng vector (sử dụng chain rule + dσ/dx = σ(1-σ)):

```
∂L/∂v_c = (σ(v_c·v_p) - 1) × v_p + Σ σ(v_c·v_nk) × v_nk

∂L/∂v_p = (σ(v_c·v_p) - 1) × v_c

∂L/∂v_nk = σ(v_c·v_nk) × v_c    (cho mỗi negative k)
```

### 9.2. Giải thích trực quan gradient

**∂L/∂v_c:**
- Positive part: `(σ-1) × v_p` — nếu σ < 1 (dot_pos chưa đủ lớn) → gradient âm → kéo v_c **về phía** v_p
- Negative part: `σ × v_nk` — nếu σ > 0 (chưa đủ xa negative) → gradient dương → đẩy v_c **xa khỏi** v_nk

**∂L/∂v_p:**
- `(σ-1) × v_c` — kéo v_p về phía v_c

**∂L/∂v_nk:**
- `σ × v_c` — đẩy v_nk xa khỏi v_c

### 9.3. Tính gradient cụ thể

**Gradient cho center vector (v_c = W[Sữa]):**

```
∂L/∂v_c = (σ(dot_pos) - 1) × v_p + σ(dot_neg1) × v_n1 + σ(dot_neg2) × v_n2

Thành phần 1 (positive): (0.5125 - 1) × [-0.30, 0.20, 0.40]
                        = -0.4875 × [-0.30, 0.20, 0.40]
                        = [0.1463, -0.0975, -0.1950]

Thành phần 2 (neg 1): σ(-0.08) × [0.40, 0.30, -0.20]
                     = 0.4800 × [0.40, 0.30, -0.20]
                     = [0.1920, 0.1440, -0.0960]

Thành phần 3 (neg 2): σ(-0.02) × [-0.10, 0.50, 0.30]
                     = 0.4950 × [-0.10, 0.50, 0.30]
                     = [-0.0495, 0.2475, 0.1485]

∂L/∂v_c = [0.1463 + 0.1920 + (-0.0495),
           -0.0975 + 0.1440 + 0.2475,
           -0.1950 + (-0.0960) + 0.1485]
        = [0.2888, 0.2940, -0.1425]
```

**Gradient cho positive context vector (v_p = W'[Bánh_mì]):**

```
∂L/∂v_p = (σ(dot_pos) - 1) × v_c
        = -0.4875 × [0.10, -0.20, 0.30]
        = [-0.0488, 0.0975, -0.1463]
```

**Gradient cho negative 1 vector (v_n1 = W'[Bơ]):**

```
∂L/∂v_n1 = σ(dot_neg1) × v_c
         = 0.4800 × [0.10, -0.20, 0.30]
         = [0.0480, -0.0960, 0.1440]
```

**Gradient cho negative 2 vector (v_n2 = W'[Mứt]):**

```
∂L/∂v_n2 = σ(dot_neg2) × v_c
         = 0.4950 × [0.10, -0.20, 0.30]
         = [0.0495, -0.0990, 0.1485]
```

---

## 10. Bước 7: Cập nhật Embedding

### 10.1. Công thức Gradient Descent

```
θ_mới = θ_cũ - η × ∂L/∂θ
```

Trong đó η (learning rate) thường = 0.01-0.05.

### 10.2. Update center embedding (W[Sữa])

Với η = 0.1:

```
W_mới[0] = W_cũ[0] - 0.1 × ∂L/∂v_c
         = [0.10, -0.20, 0.30] - 0.1 × [0.2888, 0.2940, -0.1425]
         = [0.10, -0.20, 0.30] - [0.0289, 0.0294, -0.0143]
         = [0.0711, -0.2294, 0.3143]
```

### 10.3. Update positive context embedding (W'[Bánh_mì])

```
W'_mới[1] = W'_cũ[1] - 0.1 × ∂L/∂v_p
          = [-0.30, 0.20, 0.40] - 0.1 × [-0.0488, 0.0975, -0.1463]
          = [-0.30, 0.20, 0.40] - [-0.0049, 0.0098, -0.0146]
          = [-0.2951, 0.1902, 0.4146]
```

### 10.4. Update negative 1 context embedding (W'[Bơ])

```
W'_mới[3] = W'_cũ[3] - 0.1 × ∂L/∂v_n1
          = [0.40, 0.30, -0.20] - 0.1 × [0.0480, -0.0960, 0.1440]
          = [0.40, 0.30, -0.20] - [0.0048, -0.0096, 0.0144]
          = [0.3952, 0.3096, -0.2144]
```

### 10.5. Update negative 2 context embedding (W'[Mứt])

```
W'_mới[4] = W'_cũ[4] - 0.1 × ∂L/∂v_n2
          = [-0.10, 0.50, 0.30] - 0.1 × [0.0495, -0.0990, 0.1485]
          = [-0.10, 0.50, 0.30] - [0.0050, -0.0099, 0.0149]
          = [-0.1050, 0.5099, 0.2851]
```

### 10.6. Phân tích sự thay đổi

| Vector | Trước | Sau | Hướng dịch chuyển |
|--------|-------|-----|-------------------|
| W[Sữa] | [0.10, -0.20, 0.30] | [0.07, -0.23, 0.31] | Gần Bánh_mì hơn, xa Bơ và Mứt hơn |
| W'[Bánh_mì] | [-0.30, 0.20, 0.40] | [-0.30, 0.19, 0.41] | Gần Sữa hơn |
| W'[Bơ] | [0.40, 0.30, -0.20] | [0.40, 0.31, -0.21] | Xa Sữa hơn |
| W'[Mứt] | [-0.10, 0.50, 0.30] | [-0.11, 0.51, 0.29] | Xa Sữa hơn |

---

## 11. Bước 8: Lặp lại qua nhiều Epoch

### 11.1. Một epoch là gì?

1 epoch = duyệt qua **tất cả training pairs** 1 lần.

Với 264 triệu pairs, 1 epoch có thể mất vài phút đến vài chục phút tùy hardware.

### 11.2. Quá trình hội tụ

```
Epoch 1: Loss ≈ 2.0  — vector bắt đầu dịch chuyển
Epoch 2: Loss ≈ 1.5  — Sữa gần Bánh_mì hơn 1 chút
Epoch 3: Loss ≈ 1.2  — Các cụm bắt đầu hình thành
...
Epoch 10: Loss ≈ 0.3 — Embedding ổn định
```

### 11.3. Trực quan hóa (không gian 2D)

```
Trước train:                    Sau 10 epoch:
                                
Bơ •                            Bơ •──• Mứt
    |                               |
    |   • Trứng                     |   • Trứng
    |                               |
----+------- Mứt •     ----+-------+
    |                               |
Sữa •   • Bánh_mì           Sữa •──• Bánh_mì
    |                               |
```

Sữa và Bánh_mì gần nhau (hay mua chung). Bơ và Mứt cũng gần nhau. Các cụm sản phẩm liên quan hình thành.

---

## 12. Bước 9: Tính Similarity & Recommend

### 12.1. Lấy embedding cuối cùng

Sau khi train, mỗi sản phẩm có 2 embeddings:
- **W[i]** — embedding khi là center
- **W'[i]** — embedding khi là context

Thường dùng **W[i]** làm embedding cuối, hoặc lấy trung bình:

```python
final_embedding[i] = (W[i] + W'[i]) / 2
```

### 12.2. Cosine Similarity

```
cosine(A, B) = (v_A · v_B) / (|v_A| × |v_B|)
             = Σ(v_A[i] × v_B[i]) / sqrt(Σ v_A[i]²) × sqrt(Σ v_B[i]²)
```

Giá trị từ -1 đến 1:
- 1: cùng hướng (rất giống nhau)
- 0: không liên quan
- -1: ngược hướng (rất khác nhau)

### 12.3. Ví dụ tính similarity

```python
v_Sữa     = [0.23, -0.45, 0.12, 0.78, ...]
v_Bánh_mì = [0.21, -0.40, 0.15, 0.80, ...]
v_Gạo     = [-0.50, 0.30, 0.80, -0.20, ...]

cosine(Sữa, Bánh_mì) = 0.92  ← rất gần
cosine(Sữa, Gạo)     = 0.12  ← xa
```

### 12.4. Recommend

Khi user mua sản phẩm X:
1. Lấy embedding vector của X
2. Tính cosine similarity với tất cả sản phẩm khác
3. Bỏ qua chính X
4. Sort giảm dần, lấy top-K

```
Top-5 sản phẩm tương tự Sữa nhất:
1. Bánh_mì   (0.92)
2. Trứng     (0.89)
3. Bơ        (0.85)
4. Sữa_chua  (0.78)
5. Mứt       (0.72)
```

---

## 13. Ví dụ đầy đủ bằng số

### 13.1. Dữ liệu

```
5 sản phẩm: Sữa(0), Bánh_mì(1), Trứng(2), Bơ(3), Mứt(4)
3 đơn hàng:
  Order 1: [0, 1, 2]  — Sữa, Bánh_mì, Trứng
  Order 2: [0, 1, 3]  — Sữa, Bánh_mì, Bơ
  Order 3: [2, 3, 0]  — Trứng, Bơ, Sữa
```

### 13.2. Thông số

```
D = 2 (cho dễ vẽ)
K = 1 negative sample
η = 0.1
window = 2
```

### 13.3. Khởi tạo

```
W = [[0.1, 0.2],   # Sữa
     [-0.1, 0.3],  # Bánh_mì
     [0.4, -0.2],  # Trứng
     [0.0, 0.1],   # Bơ
     [-0.3, 0.4]]  # Mứt

W' = [[0.2, -0.1],  # Sữa
      [-0.2, 0.4],  # Bánh_mì
      [0.3, 0.1],   # Trứng
      [-0.1, 0.2],  # Bơ
      [0.4, -0.3]]  # Mứt
```

### 13.4. Training pairs từ Order 1 [0, 1, 2]

```
(0,1): center=Sữa(0), context=Bánh_mì(1)
(0,2): center=Sữa(0), context=Trứng(2)
(1,0): center=Bánh_mì(1), context=Sữa(0)
(1,2): center=Bánh_mì(1), context=Trứng(2)
(2,0): center=Trứng(2), context=Sữa(0)
(2,1): center=Trứng(2), context=Bánh_mì(1)
```

### 13.5. Xử lý pair đầu tiên (0,1) — (Sữa, Bánh_mì)

**Chọn negative:** unigram distribution, giả sử chọn Mứt(4)

**Forward:**
```
v_c = W[0] = [0.1, 0.2]
v_p = W'[1] = [-0.2, 0.4]
v_n = W'[4] = [0.4, -0.3]

dot_pos = 0.1×(-0.2) + 0.2×0.4 = -0.02 + 0.08 = 0.06
dot_neg = 0.1×0.4 + 0.2×(-0.3) = 0.04 - 0.06 = -0.02

σ(0.06) = 0.515
σ(0.02) = 0.505

Loss = -log(0.515) - log(0.505) = 0.664 + 0.683 = 1.347
```

**Backward:**
```
∂L/∂v_c = (0.515-1)×[-0.2, 0.4] + 0.505×[0.4, -0.3]
        = -0.485×[-0.2, 0.4] + 0.505×[0.4, -0.3]
        = [0.097, -0.194] + [0.202, -0.152]
        = [0.299, -0.346]

∂L/∂v_p = (0.515-1)×[0.1, 0.2] = -0.485×[0.1, 0.2] = [-0.049, -0.097]

∂L/∂v_n = 0.505×[0.1, 0.2] = [0.051, 0.101]
```

**Update:**
```
W_mới[0] = [0.1, 0.2] - 0.1×[0.299, -0.346] = [0.070, 0.235]
W'_mới[1] = [-0.2, 0.4] - 0.1×[-0.049, -0.097] = [-0.195, 0.410]
W'_mới[4] = [0.4, -0.3] - 0.1×[0.051, 0.101] = [0.395, -0.310]
```

### 13.6. Sau 1 epoch (tất cả pairs từ 3 đơn hàng)

```
W[Sữa]     = [0.08, 0.22]  (gần Bánh_mì hơn)
W[Bánh_mì] = [-0.09, 0.29] (gần Sữa hơn)
W[Trứng]   = [0.38, -0.18] (gần Sữa hơn)
W[Bơ]      = [0.01, 0.09]  (gần Sữa hơn)
W[Mứt]     = [-0.31, 0.41] (xa Sữa hơn)
```

### 13.7. Sau 10 epoch

```
W[Sữa]     = [0.15, 0.30]
W[Bánh_mì] = [0.12, 0.28]  (rất gần Sữa)
W[Trứng]   = [0.10, 0.25]  (gần Sữa)
W[Bơ]      = [0.08, 0.20]  (gần Sữa)
W[Mứt]     = [-0.40, 0.50] (xa Sữa)
```

**Cosine similarity:**
```
cosine(Sữa, Bánh_mì) = 0.98
cosine(Sữa, Trứng)   = 0.95
cosine(Sữa, Bơ)      = 0.92
cosine(Sữa, Mứt)     = -0.30
```

→ Gợi ý khi mua Sữa: Bánh_mì > Trứng > Bơ

---

## 14. Code minh họa với Gensim

### 14.1. Cài đặt

```bash
pip install gensim pandas pyarrow
```

### 14.2. Code đầy đủ

```python
import pandas as pd
from gensim.models import Word2Vec

# 1. Load dữ liệu
order_products = pd.read_parquet("data/order_products.parquet")
products = pd.read_parquet("data/products.parquet")

# 2. Gom sản phẩm theo đơn hàng
#    Mỗi đơn hàng là 1 list product_id (dạng string)
orders = (
    order_products
    .groupby('order_id')['product_id']
    .apply(lambda x: [str(pid) for pid in x])
    .tolist()
)

print(f"Số đơn hàng: {len(orders)}")
print(f"VD 1 đơn: {orders[0][:5]}...")  # 5 sản phẩm đầu

# 3. Train Item2Vec
model = Word2Vec(
    sentences=orders,        # list of lists — mỗi list là 1 "câu"
    vector_size=128,         # D = 128 chiều
    window=10,               # max distance giữa center và context
    min_count=2,             # bỏ sản phẩm xuất hiện < 2 lần
    sg=1,                    # 1 = Skip-gram, 0 = CBOW
    negative=10,             # K = 10 negative samples
    epochs=10,               # số lần duyệt qua dữ liệu
    workers=4,               # số thread
    alpha=0.025,             # learning rate ban đầu
    min_alpha=0.0001,        # learning rate tối thiểu
)

# 4. Lưu model
model.save("models/item2vec/item2vec.model")

# 5. Lấy embedding của 1 sản phẩm
product_id = "101"
embedding = model.wv[product_id]
print(f"Embedding của sản phẩm {product_id}: {embedding[:5]}...")  # 5 chiều đầu

# 6. Tìm sản phẩm tương tự
similar = model.wv.most_similar(product_id, top_k=5)
print(f"\nTop-5 sản phẩm tương tự sản phẩm {product_id}:")
for pid, score in similar:
    pname = products[products['product_id'] == int(pid)]['product_name'].values
    pname = pname[0] if len(pname) else "?"
    print(f"  {pid}: {pname} (score={score:.4f})")

# 7. Tính similarity giữa 2 sản phẩm cụ thể
sim = model.wv.similarity("101", "102")
print(f"\nSimilarity giữa 101 và 102: {sim:.4f}")
```

### 14.3. Giải thích tham số Gensim

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `sentences` | list of lists | Dữ liệu đầu vào, mỗi list là 1 đơn hàng |
| `vector_size` | 128 | Số chiều embedding |
| `window` | 10 | Kích thước window (context window) |
| `min_count` | 2 | Bỏ sản phẩm xuất hiện < 2 lần |
| `sg` | 1 | 1=Skip-gram, 0=CBOW |
| `negative` | 10 | Số negative samples (K) |
| `epochs` | 10 | Số lần duyệt dữ liệu |
| `workers` | 4 | Số thread song song |
| `alpha` | 0.025 | Learning rate ban đầu |
| `min_alpha` | 0.0001 | Learning rate tối thiểu (giảm dần) |

---

## 15. Item2Vec vs Item-CF (Ochiai)

### 15.1. So sánh tổng quan

| Tiêu chí | Item2Vec | Item-CF (Ochiai) |
|----------|----------|-----------------|
| **Cách học** | Neural network (gradient descent) | Đếm co-occurrence + công thức |
| **Đầu ra** | Embedding vector dense (128D) | Ma trận similarity thưa |
| **Tham số học** | Có (W, W' — hàng triệu tham số) | Không (chỉ đếm) |
| **Thứ tự items** | Có quan tâm (window) | Không (coi như set) |
| **Quan hệ gián tiếp** | ✅ Học được (A→C→B) | ❌ Không (chỉ A→B trực tiếp) |
| **Dữ liệu cần** | Nhiều (> 1M orders) | Ít hơn |
| **Tốc độ train** | Chậm (epochs, gradient) | Nhanh (1 pass đếm) |
| **Interpretability** | Khó (black box) | Dễ (công thức rõ ràng) |
| **Cold-start** | ❌ Cần retrain | ❌ Cần cập nhật co-occurrence |

### 15.2. Ví dụ quan hệ gián tiếp

Giả sử:
- Sữa và Bánh_mì: cùng đơn 100 lần
- Sữa và Bơ: cùng đơn 80 lần
- Bánh_mì và Bơ: cùng đơn 0 lần (chưa từng mua chung)

**Item-CF:**
```
similarity(Sữa, Bánh_mì) = 0.8
similarity(Sữa, Bơ) = 0.6
similarity(Bánh_mì, Bơ) = 0.0  ← không biết gì
```

**Item2Vec:**
```
similarity(Sữa, Bánh_mì) = 0.8
similarity(Sữa, Bơ) = 0.6
similarity(Bánh_mì, Bơ) = 0.4  ← học được từ "cầu nối" Sữa
```

### 15.3. Công thức

| Model | Công thức similarity |
|-------|---------------------|
| **Item-CF (Ochiai)** | `score(A→B) = [cnt/√(cntA×cntB)] × [cnt/cntA] × log1p(cnt)` |
| **Item2Vec** | `similarity(A,B) = cosine(W[A], W[B])` |

### 15.4. Khi nào dùng cái gì?

| Tình huống | Item-CF | Item2Vec |
|------------|---------|----------|
| Dữ liệu ít (< 100K orders) | ✅ Tốt hơn | ❌ Không đủ để học |
| Dữ liệu nhiều (> 1M orders) | ✅ Vẫn tốt | ✅ Tốt hơn |
| Cần giải thích kết quả | ✅ Dễ | ❌ Khó |
| Cần real-time update | ✅ Nhanh | ❌ Phải retrain |
| Quan hệ gián tiếp | ❌ Không học được | ✅ Học được |

---

## 16. Ưu điểm & Nhược điểm

### 16.1. Ưu điểm

1. **Học được quan hệ gián tiếp:** Có thể suy luận A liên quan B dù chưa từng xuất hiện cùng đơn, thông qua sản phẩm trung gian C.

2. **Embedding dense:** Vector 128 chiều chứa nhiều thông tin hơn ma trận thưa, có thể dùng cho các tác vụ khác (phân cụm, visualization, feature cho model khác).

3. **Không cần feature engineering:** Chỉ cần dữ liệu order-product, không cần thông tin sản phẩm (danh mục, mô tả...).

4. **Có thứ tự:** Nếu thứ tự sản phẩm trong đơn có ý nghĩa (VD: mua trước/sau), Item2Vec tận dụng được.

### 16.2. Nhược điểm

1. **Cần nhiều dữ liệu:** Với < 100K đơn hàng, embedding học không tốt.

2. **Window là giả định không phù hợp:** Trong đơn hàng, thứ tự sản phẩm thường ngẫu nhiên, không có ý nghĩa ngữ pháp như câu văn. Sản phẩm quan trọng (hay mua kèm) có thể nằm ngoài window.

3. **Chậm train:** Cần nhiều epoch, nhiều pairs → thời gian train lâu.

4. **Không interpretable:** Không thể giải thích tại sao 2 sản phẩm giống nhau (khác với Item-CF có công thức rõ ràng).

5. **Cold-start:** Sản phẩm mới không có embedding → không thể recommend.

6. **Hyperparameters:** Phải tuning nhiều tham số (vector_size, window, negative, epochs...).

---

## 17. Các câu hỏi thường gặp

### Q1: Item2Vec có giống Item-CF không?

**Không.** Item2Vec dùng neural network học embedding, Item-CF đếm co-occurrence và tính similarity bằng công thức. Chúng hoàn toàn khác nhau về cách hoạt động.

### Q2: Tại sao cần 2 ma trận embedding W và W'?

Vì mỗi sản phẩm đóng 2 vai trò: center và context. Tách riêng giúp model học được 2 góc nhìn khác nhau, tăng khả năng biểu diễn.

### Q3: Tại sao dùng negative sampling thay vì softmax?

Softmax với 36,181 classes quá chậm. Negative sampling chỉ tính K+1 classes (K=5-20) → nhanh hơn nhiều.

### Q4: Window có vấn đề gì?

Nếu 2 sản phẩm hay mua kèm nhưng nằm xa nhau trong đơn hàng dài, window nhỏ sẽ bỏ sót. Giải pháp: tăng window hoặc dùng Item-CF (không quan tâm thứ tự).

### Q5: Làm sao biết model đã học tốt?

Kiểm tra:
- Loss giảm dần qua các epoch
- Sản phẩm cùng danh mục có similarity cao
- Sản phẩm hay mua chung có similarity cao
- Evaluation metrics (precision, recall, NDCG) trên test set

### Q6: Item2Vec có trong dự án này không?

**Không.** Dự án này dùng Item-CF (Ochiai) ở bước 3, không phải Item2Vec. Item2Vec là thuật toán riêng biệt.

### Q7: Tại sao gọi là "Item2Vec"?

Vì nó là **Word2Vec áp dụng cho Item**:
- "Item" = sản phẩm
- "2" = to
- "Vec" = Vector

---

## Tài liệu tham khảo

1. Barkan, O., & Koenigstein, N. (2016). *Item2Vec: Neural Item Embedding for Collaborative Filtering*. IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP).
2. Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.
3. Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NIPS 2013.
4. Gensim Documentation: *https://radimrehurek.com/gensim/models/word2vec.html*