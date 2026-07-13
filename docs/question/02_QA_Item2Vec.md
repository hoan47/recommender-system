# Bộ Câu Hỏi & Trả Lời — Item2Vec (Neural Item-Based CF)
## Word2Vec Skip-gram trên Giỏ Hàng

---

## 📖 Thuật ngữ tiếng Anh chuyên ngành (Glossary)

| # | Thuật ngữ | Phiên âm (IPA) | Giải thích | Liên hệ dự án |
|---|-----------|---------------|------------|---------------|
| 1 | **Word2Vec** | /wɜːd tuː vek/ | Kỹ thuật học embedding cho từ, do Google phát triển | Dùng Gensim Word2Vec để học embedding sản phẩm (`item_cf_neural.py` dòng 81–92) |
| 2 | **Skip-gram** | /skɪp ɡræm/ | Kiến trúc Word2Vec: dự đoán context từ target word | `sg=1` (`item_cf_neural.py` dòng 89) |
| 3 | **CBOW** | /siː bəʊ/ | Continuous Bag of Words: dự đoán target word từ context | `sg=0` — không dùng, chọn Skip-gram vì tốt hơn cho dữ liệu ít |
| 4 | **Negative Sampling** | /ˈneɡ.ə.tɪv ˈsɑːm.pəl.ɪŋ/ | Kỹ thuật huấn luyện: chỉ cập nhật 1 positive + K negative samples | `negative=10` (`item_cf_neural.py` dòng 86) |
| 5 | **Embedding** | /ɪmˈbed.ɪŋ/ | Vector số thực biểu diễn ý nghĩa sản phẩm trong không gian latent | `vector_size=128` (`item_cf_neural.py` dòng 83) |
| 6 | **Window Size** | /ˈwɪn.dəʊ saɪz/ | Số từ lân cận xung quanh target word để xét context | `window=5` (`item_cf_neural.py` dòng 84) |
| 7 | **Min Count** | /mɪn kaʊnt/ | Ngưỡng lọc: từ xuất hiện dưới ngưỡng sẽ bị bỏ qua | `min_count=10` (`item_cf_neural.py` dòng 85) |
| 8 | **Epochs** | /ˈiː.pɒks/ | Số lần lặp qua toàn bộ dữ liệu huấn luyện | `epochs=20` (`item_cf_neural.py` dòng 87) |
| 9 | **Gensim** | /ˈɡen.sɪm/ | Thư viện Python cho topic modeling và word embedding | `from gensim.models import Word2Vec` (`item_cf_neural.py` dòng 8) |
| 10 | **Cosine Similarity** | /ˈkəʊ.saɪn ˌsɪm.ɪˈlær.ɪ.ti/ | Cosine của góc giữa 2 embedding vector | `most_similar()` (`item_cf_neural.py` dòng 116) dùng cosine |
| 11 | **Neural-based** | /ˈnjʊə.rəl beɪst/ | Dùng neural network để học tham số | Item2Vec là Neural Item-Based CF |
| 12 | **Model-based** | /ˈmɒd.əl beɪst/ | Học tham số qua tối ưu hàm loss | Item2Vec là model-based, khác với Item-CF memory-based |
| 13 | **Hyperparameter** | /ˈhaɪ.pə.pəˈræm.ɪ.tər/ | Tham số cấu hình, không học từ dữ liệu | `I2V_VECTOR_SIZE`, `I2V_WINDOW`, `I2V_EPOCHS` trong `config.py` |
| 14 | **Workers** | /ˈwɜː.kəz/ | Số luồng xử lý song song | `workers=4` (`item_cf_neural.py` dòng 88) |
| 15 | **Seed** | /siːd/ | Giá trị khởi tạo random, đảm bảo tái lập kết quả | `seed=RANDOM_SEED` (`item_cf_neural.py` dòng 90) |
| 16 | **Indirect Relationship** | /ˌɪn.daɪˈrekt rɪˈleɪ.ʃən.ʃɪp/ | Quan hệ gián tiếp A→B→C → A~C | Item2Vec capture được, Item-CF không |

---

## PHẦN 1 — Thiết Kế Mô Hình

### Q1: Tại sao gọi là Item2Vec? Liên hệ với Word2Vec như thế nào?

**Trả lời:**

Item2Vec là ứng dụng của Word2Vec vào dữ liệu giỏ hàng:

| Word2Vec | Item2Vec |
|----------|----------|
| Văn bản (corpus) | Lịch sử đơn hàng |
| Câu (sentence) | 1 đơn hàng (order) |
| Từ (word) | Sản phẩm (product) |
| Embedding từ | Embedding sản phẩm |
| Từ gần nghĩa | Sản phẩm hay mua kèm nhau |
|Quan tâm tới thứ tự các từ trong 1 câu | Không quan tâm thứ tự sản phẩm trong đơn hàng |

Ý tưởng cốt lõi: các sản phẩm thường mua cùng nhau → "ngữ cảnh" giống nhau → embedding gần nhau trong không gian vector.

**Dẫn chứng code** (`item_cf_neural.py`, dòng 19–24):
```python
class ItemCFNeuralModel:
    """
    Item2Vec (Neural Item-Based CF): Word2Vec Skip-gram trên giỏ hàng.
    
    Mỗi đơn hàng là 1 "câu", mỗi sản phẩm là 1 "từ".
    Học embedding từ co-occurrence context → Neural Item-Based CF.
    """
```

---

### Q2: Em chọn Skip-gram hay CBOW? Tại sao?

**Trả lời:**

Em chọn **Skip-gram** (`sg=1`).

**Dẫn chứng code** (`item_cf_neural.py`, dòng 89):
```python
sg=1,  # Skip-gram
```

**Lý do chọn Skip-gram:**

| Tiêu chí | Skip-gram | CBOW |
|----------|-----------|------|
| Hướng dự đoán | Từ center dự đoán context | Từ context dự đoán center |
| Sản phẩm hiếm (long-tail) | **Tốt hơn** — học từng context pair riêng lẻ | Kém hơn — trung bình hóa context |
| Tốc độ train | Chậm hơn | Nhanh hơn |
| Phù hợp với Instacart | **Có**, vì ~46% sản phẩm xuất hiện < 50 lần | Kém với long-tail |

Instacart có phân phối long-tail nặng → Skip-gram học tốt hơn cho sản phẩm ít phổ biến.

---

### Q3: `window = 5` có nghĩa gì trong bối cảnh giỏ hàng?

**Trả lời:**

Trong Word2Vec gốc, window = số từ bên trái/phải được xét là "context". Trong Item2Vec, vì các sản phẩm trong 1 đơn hàng **không có thứ tự ngữ nghĩa** (thứ tự chỉ là thứ tự cho vào giỏ), window = 5 nghĩa là:

Với mỗi sản phẩm trung tâm (center), 5 sản phẩm trước và 5 sản phẩm sau được xét là "ngữ cảnh" (context).

Thực tế: giỏ hàng trung bình 10 sản phẩm, window = 5 → hầu hết mọi cặp trong giỏ đều được xét là cặp (center, context) ít nhất 1 lần.

**Dẫn chứng code** (`config.py`, dòng 61):
```python
I2V_WINDOW = 5
```

---

### Q4: Negative Sampling là gì? Tại sao cần?

**Trả lời:**

Skip-gram gốc cần cập nhật toàn bộ vocabulary (36K sản phẩm) cho mỗi cặp (center, context) → **cực kỳ chậm**.

**Negative Sampling** giải quyết: thay vì cập nhật tất cả, chỉ:
- 1 **positive pair** (center, context thật) → label = 1
- K **negative pairs** (center, context giả) → label = 0

Bài toán trở thành **binary classification** cho mỗi cặp, cập nhật đúng K+1 embeddings/bước thay vì toàn bộ.

Em dùng `negative = 10` → mỗi positive pair có 10 negative pairs.

**Dẫn chứng code** (`item_cf_neural.py`, dòng 86):
```python
negative=self.params['negative'],  # = 10
```

---

### Q5: Negative samples được chọn theo tiêu chí nào? Ngẫu nhiên hoàn toàn không?

**Trả lời:**

**Không hoàn toàn ngẫu nhiên uniform.** Word2Vec dùng **Unigram Distribution với power 3/4**:

```
P(w) ∝ freq(w)^(3/4)
```

Nghĩa là: sản phẩm phổ biến hơn có xác suất được chọn làm negative cao hơn, nhưng được **"làm phẳng"** bởi power 3/4 (giảm bias với sản phẩm cực phổ biến).

Ví dụ:
- Nước lọc (freq=100,000): `100000^0.75 ≈ 17,783`
- Sốt truffle (freq=100): `100^0.75 ≈ 31.6`
- Ratio = 17783/31.6 ≈ 563 (không phải 1000 như uniform)

→ Sản phẩm hiếm vẫn có cơ hội được chọn làm negative, giúp học embedding tốt hơn.

**Lưu ý**: Phần này được xử lý tự động bên trong Gensim, không cần code riêng.

---

## PHẦN 2 — Ma Trận W và W'

### Q6: Word2Vec có 2 ma trận W và W'. Em lấy ma trận nào làm embedding kết quả? Tại sao?

**Trả lời:**

Em lấy **ma trận W (input embedding matrix)** — tức là `model.wv` trong Gensim.

Kiến trúc Word2Vec có:
- **W** (input matrix): kích thước `(vocab_size, vector_size)` — embedding của center word
- **W'** (output matrix): kích thước `(vector_size, vocab_size)` — embedding ngữ cảnh

Em lấy W vì:
1. **Thực nghiệm**: W hội tụ tốt hơn và biểu diễn ngữ nghĩa mượt hơn
2. **Chuẩn Gensim**: `model.wv` chính là W, và là chuẩn de-facto của Item2Vec
3. **Đối xứng**: trong bài toán giỏ hàng, không có phân biệt "center" vs "context" về mặt ngữ nghĩa — nên W đủ

**Dẫn chứng code** (`item_cf_neural.py`, dòng 116):
```python
similar = self.model.wv.most_similar(pid_str, topn=top_k)
```
→ `model.wv` là ma trận W, `most_similar` tính cosine similarity trực tiếp trên W.

---

### Q7: Giá trị khởi tạo của ma trận W ban đầu là gì?

**Trả lời:**

Gensim khởi tạo W bằng **random uniform** trong khoảng `[-0.5/vector_size, 0.5/vector_size]`. Với `vector_size = 128`:
- Khoảng khởi tạo: `[-1/256, 1/256]` ≈ `[-0.00391, 0.00391]`

Lý do dùng uniform nhỏ:
- Tránh vanishing/exploding gradient khi bắt đầu
- Các embedding gần nhau lúc ban đầu → học cách phân biệt qua training

**Dẫn chứng code**: Seed được fix để reproducibility (`item_cf_neural.py`, dòng 90):
```python
seed=RANDOM_SEED,   # RANDOM_SEED = 42 (config.py)
```

---

### Q8: Loss function của Skip-gram + Negative Sampling là gì? W được cập nhật như thế nào?

**Trả lời:**

**Objective function** (maximize):
```
J = log σ(v'_context · v_center) + Σ E[log σ(-v'_neg · v_center)]
```

Trong đó `σ` là sigmoid, `v'` là hàng của W'.

**Gradient descent** (simplified):
```
∂J/∂v_center = (1 - σ(v'_pos · v_center)) × v'_pos 
             - Σ σ(v'_neg · v_center) × v'_neg
```

Mỗi bước SGD:
1. Tính sigmoid score cho positive và negative pairs
2. Tính gradient
3. Cập nhật `v_center` (hàng trong W) và các `v'_context` tương ứng (cột trong W')
4. Chỉ K+1 vectors được cập nhật mỗi bước (không phải toàn bộ vocabulary)

---

### Q9: Sau khi train xong, embedding của sản phẩm A nằm ở đâu?

**Trả lời:**

Embedding của sản phẩm A (product_id dưới dạng string) nằm tại:
```python
self.model.wv[str(product_id)]   # numpy array shape (128,)
```

Đây là hàng tương ứng của sản phẩm đó trong ma trận W.

Khi `recommend`, tính cosine similarity giữa embedding của A với tất cả sản phẩm khác:
```python
similar = self.model.wv.most_similar(pid_str, topn=top_k)
```
→ Gensim tự tính cosine similarity và sort, trả về list `(product_id_str, cosine_sim)`.

**Dẫn chứng code** (`item_cf_neural.py`, dòng 111–118):
```python
pid_str = str(product_id)
if pid_str not in self.model.wv:
    return []

try:
    similar = self.model.wv.most_similar(pid_str, topn=top_k)
    result = [(int(pid), float(sim)) for pid, sim in similar]
    return result
except KeyError:
    return []
```

---

## PHẦN 3 — Dữ Liệu & Huấn Luyện

### Q10: Dữ liệu đầu vào được chuẩn bị thế nào? Từ order_products → sentences ra sao?

**Trả lời:**

Đầu vào là DataFrame `order_products` gồm các cột `order_id`, `product_id`. Quy trình:

1. **Group by order_id:** Gom tất cả sản phẩm của cùng 1 đơn hàng
2. **Lọc sản phẩm hợp lệ:** Chỉ giữ sản phẩm có trong `product_id_to_idx`
3. **Lọc order có ≥ 2 items:** Order chỉ có 1 sản phẩm thì không có "ngữ cảnh" → bỏ qua
4. **Chuyển product_id sang string:** Gensim yêu cầu tag là string

Kết quả: ~3.3 triệu sentences, mỗi sentence là list string product_id.

**Dẫn chứng code** (`item_cf_neural.py`, dòng 56–64):
```python
print("  Đang tạo sentences từ orders...")
sentences = []
grouped = order_products.groupby('order_id')['product_id']

for order_id, group in grouped:
    items = [str(pid) for pid in group if pid in self.product_id_to_idx]
    if len(items) >= 2:  # Order phải có ít nhất 2 items
        sentences.append(items)
```

---

### Q11: Tại sao lại convert product_id sang string khi đưa vào Gensim?

**Trả lời:**

Gensim Word2Vec yêu cầu đầu vào là **list of list of strings** (corpus dạng text). Vocabulary của Gensim là `dict[str, int]`.

Product_id là integer (ví dụ: `1234`), phải convert sang string `"1234"` để:
1. Gensim accept được
2. Khi retrieve: `model.wv["1234"]` → lấy embedding

**Dẫn chứng code** (`item_cf_neural.py`, dòng 62 và 111–117):
```python
# Train:
items = [str(pid) for pid in group if pid in self.product_id_to_idx]

# Recommend:
pid_str = str(product_id)
similar = self.model.wv.most_similar(pid_str, topn=top_k)
result = [(int(pid), float(sim)) for pid, sim in similar]
```
→ Convert ngược về `int` khi trả kết quả.

---

### Q12: `min_count = 10` trong Item2Vec khác với `min_support = 10` trong Item-CF như thế nào?

**Trả lời:**

| | Item-CF `min_support` | Item2Vec `min_count` |
|-|-----------------------|----------------------|
| Lọc cái gì | **Cặp** (A,B) xuất hiện < 10 lần | **Sản phẩm** xuất hiện < 10 lần |
| Tác động | Cặp ít xuất hiện bị bỏ, sản phẩm vẫn tồn tại trong index | Sản phẩm ít xuất hiện bị loại khỏi vocabulary hoàn toàn |
| Xử lý cold-start | Trả về `[]` | Sản phẩm không có trong `model.wv` → trả về `[]` |

**Dẫn chứng code** (`config.py`, dòng 62):
```python
I2V_MIN_COUNT = 10   # Gensim loại product xuất hiện < 10 lần
```

---

### Q13: Tại sao vector_size = 128? Nếu 64 hay 256 thì sao?

**Trả lời:**

**128 là giá trị cân bằng:**
- **64:** Quá thấp → không đủ không gian để biểu diễn 36K sản phẩm
- **128:** Đủ để capture ngữ nghĩa sản phẩm, tốc độ train chấp nhận được
- **256:** Cao hơn → chất lượng có thể tăng nhẹ nhưng tốn gấp đôi bộ nhớ và thời gian

**Dẫn chứng code** (`config.py`, dòng 60):
```python
I2V_VECTOR_SIZE = 128
```

---

### Q14: epochs = 20, workers = 4 có ý nghĩa gì?

**Trả lời:**

- **epochs=20:** Duyệt qua toàn bộ 3.3M sentences 20 lần. Mỗi lần cập nhật W và W'
- **workers=4:** Chia dữ liệu thành 4 phần, xử lý song song trên 4 luồng CPU

**Nếu epochs=5:** Chưa đủ, model underfit, embedding chưa tốt
**Nếu workers=1:** Chậm hơn ~4 lần

**Dẫn chứng code** (`config.py`, dòng 64–65):
```python
I2V_EPOCHS = 20
I2V_WORKERS = 4
```

---

### Q15: LossLogger callback làm gì?

**Trả lời:**

`LossLogger` là callback được gọi sau mỗi epoch, in ra số epoch hiện tại. Giúp theo dõi tiến trình training.

**Dẫn chứng code** (`item_cf_neural.py`, dòng 72–79):
```python
class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"    Epoch {self.epoch}/{model.epochs}")
```

---

## PHẦN 4 — Inference & So Sánh

### Q16: recommend() xử lý thế nào khi sản phẩm không có trong model.wv?

**Trả lời:**

Trả về `[]` (danh sách rỗng). Code kiểm tra:

**Dẫn chứng code** (`item_cf_neural.py`, dòng 111–113):
```python
pid_str = str(product_id)
if pid_str not in self.model.wv:
    return []
```

**Nguyên nhân sản phẩm không có trong model.wv:**
1. Sản phẩm xuất hiện < min_count (10) lần → bị Gensim loại khỏi từ vựng
2. Sản phẩm mới (cold-start) chưa từng xuất hiện trong train data

---

### Q17: Item2Vec có hướng không? Tại sao?

**Trả lời:**

**Không.** Item2Vec cho similarity đối xứng: `sim(A,B) = sim(B,A)`.

Lý do: `most_similar()` dùng Cosine similarity giữa 2 vector W, và Cosine là đối xứng:
```
cos(A, B) = (A·B) / (||A|| × ||B||) = cos(B, A)
```

Đây là hạn chế của Item2Vec so với Item-CF. Trong ensemble, Item-CF (trọng số 0.5) cung cấp tính hướng, Item2Vec (0.25) chỉ bổ sung.

---

### Q18: Item2Vec capture quan hệ gián tiếp thế nào?

**Trả lời:**

Item2Vec học embedding từ context window. Nếu:
- A thường đi với B (cùng order)
- B thường đi với C (cùng order)

Thì embedding của A và C sẽ gần nhau vì chúng có context tương tự (đều gần B). Đây gọi là **quan hệ gián tiếp (indirect relationship)**.

Item-CF không làm được điều này vì nó chỉ đếm cặp trực tiếp (A,B) và (B,C), không thấy (A,C).

---

### Q19: Item2Vec so với KGMetapath?

**Trả lời:**

| Tiêu chí | Item2Vec | KGMetapath |
|----------|---------|------------|
| **Dữ liệu** | Chỉ co-occurrence | Co-occurrence + Aisle + Department |
| **Quan hệ** | Bậc 1 + gián tiếp | Bậc cao (metapath walk) |
| **Long-tail** | ⚠️ min_count=10 | ✅ semantic walk |
| **Cold-start** | ❌ | ⚠️ qua aisle |
| **Tốc độ train** | Nhanh | Chậm (walk + train) |
| **Giải thích** | Khó (hộp đen) | Trung bình (white-box walk) |

---

### Q20: Vai trò của Item2Vec trong Ensemble?

**Trả lời:**

Item2Vec đóng góp **quan hệ gián tiếp** và **ngữ nghĩa sản phẩm**:
- Item-CF (0.5): Quan hệ trực tiếp, có hướng
- **Item2Vec (0.25): Quan hệ gián tiếp, ngữ nghĩa**
- KGMetapath (0.25): Quan hệ bậc cao, long-tail

Item2Vec bổ sung những gì Item-CF thiếu: nếu A và C không trực tiếp cùng xuất hiện nhưng thường đi cùng B, Item2Vec vẫn gợi ý được C cho A.

**Dẫn chứng code** (`config.py`, dòng 88–90):
```python
ENS_ALPHA = 0.5   # Item-CF
ENS_BETA = 0.25   # Item2Vec
ENS_GAMMA = 0.25  # KGMetapath
```

---

### Q21: Item2Vec có bị substitute không? Giải pháp?

**Trả lời:**

**Có.** Item2Vec chỉ dựa trên co-occurrence, không phân biệt complementary và substitute. Nếu Coca và Pepsi thường xuất hiện cùng nhau, embedding của chúng gần nhau → gợi ý lẫn nhau.

**Giải pháp:** CB Filter loại bỏ substitute sau ensemble.

---

### Q22: Hạn chế lớn nhất của Item2Vec là gì?

**Trả lời:**

**Hạn chế lớn nhất: Không có hướng (đối xứng).**

Item2Vec không phân biệt A→B và B→A. Trong thực tế:
- "Nước lẩu → thịt bò" nên được gợi ý (conf cao)
- "Thịt bò → nước lẩu" không nên (conf thấp)
- Item2Vec cho cả 2 similarity như nhau

---

### Q23: Nếu được cải thiện Item2Vec, em sẽ làm gì?

**Trả lời:**

3 hướng cải thiện:

1. **Thêm hướng cho Item2Vec:** Dùng embedding riêng cho target và context (như trong paper Item2Vec gốc), tính score bất đối xứng

2. **Tăng min_count cho long-tail:** Dùng sub-sampling để giảm ảnh hưởng của sản phẩm quá phổ biến, giúp sản phẩm hiếm có cơ hội hơn

3. **Kết hợp thông tin danh mục:** Thêm aisle/department vào quá trình học embedding (giống KGMetapath) để hỗ trợ cold-start

**Dẫn chứng code** (`item_cf_neural.py`, dòng 81–92, code hiện tại):
```python
self.model = Word2Vec(
    sentences=sentences,
    vector_size=self.params['vector_size'],
    window=self.params['window'],
    min_count=self.params['min_count'],
    negative=self.params['negative'],
    epochs=self.params['epochs'],
    workers=self.params['workers'],
    sg=1,  # Skip-gram
    seed=RANDOM_SEED,
    callbacks=[LossLogger()]
)
```

---

## TÓM TẮT LUỒNG THUẬT TOÁN

```
order_products (31.9M records)
    │
    ▼
Group by order_id → sentences (3.3M)
    │ (item_cf_neural.py dòng 56–64)
    ▼
Word2Vec(sg=1, vector_size=128, window=5, min_count=10, negative=10, epochs=20)
    │ (item_cf_neural.py dòng 81–92)
    ├── Khởi tạo W (input embedding) và W' (output embedding) uniform random
    ├── Mỗi epoch: duyệt sentences, cập nhật W và W' qua Skip-gram + Negative Sampling
    └── Kết quả: W (model.wv) là embedding cuối cùng
    │
    ▼
recommend(product_id):
    ├── Nếu pid không trong model.wv → return []
    └── Nếu có → most_similar() = Cosine similarity trên W → top-K
        (item_cf_neural.py dòng 111–118)