# Bộ Câu Hỏi & Trả Lời — CB Filter (Content-Based Diversity Filter)
## Ensemble TF-IDF + Count Vectorizer + Multi-field

---

## 📖 Thuật ngữ tiếng Anh chuyên ngành (Glossary)

| # | Thuật ngữ | Phiên âm (IPA) | Giải thích | Liên hệ dự án |
|---|-----------|---------------|------------|---------------|
| 1 | **Content-Based Filtering (CB)** | /ˈkɒn.tent beɪst ˈfɪl.tər.ɪŋ/ | Lọc dựa trên nội dung: gợi ý dựa trên đặc tính sản phẩm (tên, aisle, department), không dùng hành vi người dùng | CB Filter dùng TF-IDF trên tên sản phẩm tiếng Việt (`vectorizer.py`) |
| 2 | **TF-IDF** | /tiː ef aɪ diː ef/ | Term Frequency - Inverse Document Frequency: đo mức độ quan trọng của từ trong văn bản | `TfidfVectorizer()` (`vectorizer.py` dòng 108–114) |
| 3 | **Count Vectorizer** | /kaʊnt ˌvek.tər.aɪˈzeɪ.ər/ | Đếm tần suất xuất hiện của từ, không normalize | `CountVectorizer()` (`vectorizer.py` dòng 142–148) |
| 4 | **N-gram** | /en ɡræm/ | Cụm n từ liên tiếp: unigram (1 từ), bigram (2 từ) | `ngram_range=(1,2)` cho tên sản phẩm (`config.py` dòng 33) |
| 5 | **Overlap Coefficient** | /ˈəʊ.və.læp ˌkəʊ.ɪˈfɪʃ.ənt/ | \|A∩B\| / min(\|A\|,\|B\|) — chỉ quan tâm tỷ lệ từ trùng so với bên ít từ hơn | Tính tại `vectorizer.py` dòng 239–245 |
| 6 | **Cosine Similarity** | /ˈkəʊ.saɪn ˌsɪm.ɪˈlær.ɪ.ti/ | Cosine của góc giữa 2 vector | Dùng cho nhánh TF-IDF: `sim_tfidf` (`vectorizer.py` dòng 223) |
| 7 | **Multi-field** | /ˈmʌl.ti fiːld/ | Nhiều trường dữ liệu: tên sản phẩm, aisle, department | `build_multi_field_vectors()` (`vectorizer.py` dòng 156–212) |
| 8 | **Preprocessing / Text Cleaning** | /priːˈprəʊ.ses.ɪŋ/ | Làm sạch văn bản: lowercase, xóa số đo, stopwords, ký tự đặc biệt | `_clean_text_preprocessor()` (`vectorizer.py` dòng 78–99) |
| 9 | **Stopwords** | /ˈstɒp.wɜːdz/ | Từ dừng: từ phổ biến nhưng ít ý nghĩa (và, thì, của, các...) | File `vietnamese_stopwords.txt`, xóa tại `vectorizer.py` dòng 90–91 |
| 10 | **On-demand / Inference-time** | /ɒn dɪˈmɑːnd/ | Chỉ tính toán khi cần, không precompute trước | CB chỉ tính similarity cho candidate đã được đề xuất (`cb_filter.py` dòng 179–182) |
| 11 | **Cold-start** | /kəʊld stɑːt/ | Sản phẩm mới chưa có dữ liệu | CB xử lý cold-start: giữ nguyên candidates nếu không có vector (`cb_filter.py` dòng 153–155) |
| 12 | **Substitute** | /ˈsʌb.stɪ.tjuːt/ | Sản phẩm thay thế (Coca vs Pepsi) | CB Filter loại bỏ substitute khi similarity ≥ threshold (`cb_filter.py` dòng 187) |
| 13 | **Complementary** | /ˌkɒm.plɪˈmen.tər.i/ | Sản phẩm mua kèm (bia + snack) | CB Filter giữ lại complementary khi similarity < threshold |
| 14 | **Threshold** | /ˈθreʃ.həʊld/ | Ngưỡng quyết định substitute vs complementary | `ENS_CB_THRESHOLD = 0.25` (`config.py` dòng 93) |
| 15 | **Sparse Matrix** | /spɑːs ˈmeɪ.trɪks/ | Ma trận thưa: hầu hết phần tử bằng 0 | TF-IDF output là CSR sparse (`vectorizer.py` dòng 210) |
| 16 | **HStack** | /eɪtʃ stæk/ | Ghép ngang (horizontal stack) các ma trận thưa | `sparse_hstack(vectors, format='csr')` (`vectorizer.py` dòng 210) |
| 17 | **Diversity Filter** | /daɪˈvɜː.sɪ.ti ˈfɪl.tər/ | Bộ lọc đa dạng: loại bỏ sản phẩm quá giống nhau | CB Filter là diversity filter, loại substitute khỏi kết quả gợi ý |
| 18 | **Post-processing** | /pəʊst ˈprəʊ.ses.ɪŋ/ | Xử lý sau khi model chính đã chạy | CB là tầng hậu xử lý, áp dụng sau ensemble |

---

## PHẦN 1 — Mục Đích và Vai Trò

### Q1: CB Filter có vai trò gì trong hệ thống? Tại sao gọi là "Diversity Filter"?

**Trả lời:**

CB Filter là bộ lọc **hậu xử lý** (post-processing), chạy **sau** khi Ensemble model đã gợi ý top-K candidates. Nó loại bỏ các sản phẩm **substitute** (thay thế — quá giống nhau) để giữ lại **complementary** (bổ sung — khác nhau nhưng mua kèm được).

**Ví dụ:**
- Nếu user nhìn sản phẩm A = "Sữa tươi nguyên chất Vinamilk 1L"
- Candidate B = "Sữa tươi nguyên chất TH True Milk 1L" → **substitute** → loại bỏ
- Candidate C = "Ngũ cốc ăn sáng Kellogg's" → **complementary** → giữ lại

Không có CB Filter, co-occurrence model thường gợi ý nhiều sản phẩm cùng loại (vì người dùng thích sữa thường mua nhiều loại sữa khác nhau trong các lần mua khác nhau → co-occurrence cao nhưng không phải "mua kèm").

**Dẫn chứng code** (`cb_filter.py`, dòng 1–4):
```python
"""
Content-Based Diversity Filter.
Bộ lọc hậu xử lý — loại bỏ substitute (sản phẩm thay thế/quá giống)
khỏi kết quả gợi ý của các model co-occurrence.
"""
```

---

### Q2: CB Filter dùng thông tin gì để xác định 2 sản phẩm giống nhau?

**Trả lời:**

CB Filter dùng **nội dung văn bản** của sản phẩm — không dùng hành vi mua hàng. Cụ thể là 3 trường tiếng Việt:

1. **`product_name`** — tên sản phẩm (quan trọng nhất, weight = 1.0)
2. **`aisle`** — lối đi / danh mục con (weight = 0.8)
3. **`department`** — phòng ban / danh mục lớn (weight = 0.6)

Sử dụng kết hợp 2 bộ vector hóa:
- **TF-IDF multi-field** — vector hóa và kết hợp 3 trường theo trọng số
- **Count Vectorizer** — chỉ trên `product_name`, tính Overlap Coefficient

**Dẫn chứng code** (`cb_filter.py`, dòng 96–118):
```python
# 1. TF-IDF multi-field: name + aisle + department (có trọng số)
fields_dict = {
    'name': {
        'texts': text_name,
        'weight': CB_NAME_WEIGHT,       # 1.0
        'ngram_range': ngram_range,
        'max_features': max_features,
    },
    'aisle': {
        'texts': text_aisle,
        'weight': CB_AISLE_WEIGHT,      # 0.8
        'ngram_range': CB_AISLE_N_GRAM_RANGE,
        'max_features': CB_AISLE_MAX_FEATURES,
    },
    'dept': {
        'texts': text_dept,
        'weight': CB_DEPT_WEIGHT,       # 0.6
        'ngram_range': CB_DEPT_N_GRAM_RANGE,
        'max_features': CB_DEPT_MAX_FEATURES,
    },
}
self.product_vectors_tfidf, self._vectorizers = build_multi_field_vectors(fields_dict)
```

---

### Q3: Tại sao CB là post-processing mà không phải model chính?

**Trả lời:**

CB không thể làm model gợi ý chính vì:

1. **Chỉ dựa trên nội dung, không dùng hành vi:** CB chỉ biết tên sản phẩm giống nhau, không biết sản phẩm nào thường được mua cùng nhau. Ví dụ: "Bia" và "Snack" có tên hoàn toàn khác nhau → similarity = 0 → CB không thể biết chúng nên được gợi ý cùng nhau.

2. **Không capture được mối quan hệ mua kèm:** Mua kèm là quan hệ hành vi, không phải quan hệ nội dung. "Thịt bò" và "Nước lẩu" có tên khác xa nhau nhưng thường được mua cùng nhau.

3. **Vai trò đúng là bộ lọc:** CB phát huy sức mạnh khi loại bỏ substitute — thứ mà các model co-occurrence (Item-CF, Item2Vec) không làm được.

---

## PHẦN 2 — TF-IDF Multi-field

### Q4: Ma trận TF-IDF cuối cùng có cấu trúc như thế nào?

**Trả lời:**

3 ma trận TF-IDF riêng lẻ được scale theo trọng số rồi **ghép ngang (hstack)**:

```
TF-IDF_combined = [TF-IDF_name × 1.0 | TF-IDF_aisle × 0.8 | TF-IDF_dept × 0.6]
Shape: (n_products, features_name + features_aisle + features_dept)
     = (36181, 15000 + 500 + 100)
     = (36181, 15600)
```

Mỗi hàng = 1 sản phẩm, biểu diễn bởi vector 15,600 chiều trong không gian n-gram TF-IDF đa trường.

**Dẫn chứng code** (`vectorizer.py`, dòng 199–211):
```python
matrix = tfidf.fit_transform(texts)
if weight != 1.0:
    matrix = (matrix * weight).tocsr()  # Scale theo trọng số
vectors.append(matrix)

combined = sparse_hstack(vectors, format='csr')  # Ghép ngang
```

---

### Q5: TF-IDF được tính như thế nào? Giải thích TF và IDF.

**Trả lời:**

**TF (Term Frequency)** = tần suất từ trong document (tên sản phẩm):
```
TF(t, d) = count(t in d) / total_terms(d)
```

**IDF (Inverse Document Frequency)** = nghịch đảo tần suất từ qua toàn corpus:
```
IDF(t) = log((1 + n) / (1 + df(t))) + 1
```
Trong đó `n` = tổng số sản phẩm, `df(t)` = số sản phẩm chứa từ t.

**Ý nghĩa**: Từ xuất hiện ở nhiều sản phẩm (như "hữu cơ", "tươi") → IDF thấp → ít phân biệt. Từ đặc trưng riêng (như "quinoa", "matcha") → IDF cao → phân biệt tốt.

TF-IDF = TF × IDF, sau đó **L2-normalize** mỗi document vector → cosine similarity = dot product.

Sklearn `TfidfVectorizer` tự động xử lý tất cả điều này.

---

### Q6: N-gram range `(1, 2)` nghĩa là gì? Cho ví dụ.

**Trả lời:**

N-gram range `(1, 2)` = dùng cả **unigram** (từ đơn) và **bigram** (cặp từ liên tiếp).

Ví dụ tên sản phẩm: `"sữa chua nguyên chất Vinamilk"`

Unigrams: `["sữa", "chua", "nguyên", "chất", "vinamilk"]`
Bigrams: `["sữa chua", "chua nguyên", "nguyên chất", "chất vinamilk"]`

Tại sao cần bigram?
- `"sữa chua"` là 1 khái niệm khác với `"sữa"` + `"chua"` riêng lẻ
- `"dầu ô liu"` vs `"dầu dừa"` — unigram `"dầu"` giống nhau nhưng bigram phân biệt được

**Dẫn chứng code** (`config.py`, dòng 33, 40–41):
```python
CB_N_GRAM_RANGE = (1, 2)    # TF-IDF: word 1-gram đến 2-gram
CB_AISLE_N_GRAM_RANGE = (1, 1)  # Aisle ngắn → chỉ unigram
CB_DEPT_N_GRAM_RANGE = (1, 1)   # Department ngắn → chỉ unigram
```

---

### Q7: Tại sao dùng tiếng Việt thay vì tiếng Anh gốc? Ảnh hưởng gì đến TF-IDF?

**Trả lời:**

Dữ liệu gốc Instacart là tiếng Anh. Em đã dịch toàn bộ tên sản phẩm, aisle, department sang tiếng Việt (file `products_vi.csv`, `aisles_vi.csv`, `departments_vi.csv`).

**Lý do dùng tiếng Việt cho TF-IDF:**
1. Sản phẩm cùng loại trong tiếng Việt có từ đặc trưng hơn ("sữa tươi" vs "sữa chua" vs "sữa đặc")
2. Tránh bias từ thương hiệu Anh (Organic Valley, Horizon) vào TF-IDF
3. Phù hợp với vietnamese_stopwords.txt để lọc stopwords

**Trong vectorizer:** Có bộ tiền xử lý `_clean_text_preprocessor` xử lý riêng cho tiếng Việt:
- Lowercase
- Xóa đơn vị đo lường (mg, oz, ml, ...)
- Xóa stopwords tiếng Việt (bằng regex đã compile sẵn)
- Xóa ký tự đặc biệt

**Dẫn chứng code** (`vectorizer.py`, dòng 78–99):
```python
def _clean_text_preprocessor(text):
    text = text.lower()
    text = _PATTERN_CLEAN.sub("", text)        # Xóa đơn vị đo lường
    text = _REGEX_SPECIAL_STOPWORDS.sub('', text)
    text = _REGEX_WORD_STOPWORDS.sub('', text)  # Xóa stopwords
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## PHẦN 3 — Count Vectorizer & Ensemble Similarity

### Q8: Count Vectorizer dùng để làm gì? Khác TF-IDF như thế nào?

**Trả lời:**

Count Vectorizer đếm tần suất từ đơn thuần (**không chia cho IDF**):
```
Count[t, d] = số lần từ t xuất hiện trong document d
```

Sau đó tính **Overlap Coefficient** (không phải cosine):
```
Overlap(A, B) = |terms(A) ∩ terms(B)| / min(|terms(A)|, |terms(B)|)
```

| | TF-IDF | Count + Overlap |
|-|--------|-----------------|
| Từ thường (stopword đã lọc) | Trọng số thấp (IDF nhỏ) | Đếm bình đẳng (nếu qua stopword filter) |
| Phát hiện tên trùng nhau | Ít hiệu quả (bị IDF pha loãng) | Tốt — đếm tuyến tính số từ trùng |
| Từ hiếm đặc trưng | Trọng số cao (IDF lớn) | Ít ảnh hưởng |

→ Count + Overlap tốt cho việc phát hiện **tên sản phẩm gần giống nhau** (cùng brand, chỉ khác dung tích/hương vị).

**Dẫn chứng code** (`vectorizer.py`, dòng 239–245):
```python
# Overlap Coefficient: |A∩B| / min(|A|, |B|)
min_lengths = np.minimum(sum_a, sum_b).ravel()
sim_count = np.divide(intersection, min_lengths,
                      out=np.zeros_like(intersection, dtype=float),
                      where=min_lengths != 0)
```

---

### Q9: `alpha = 0.5` nghĩa là gì trong Ensemble similarity?

**Trả lời:**

Ensemble similarity kết hợp 2 nhánh:

```
final_sim = alpha × sim_count + (1 - alpha) × sim_tfidf
           = 0.5 × overlap_coefficient + 0.5 × cosine_tfidf
```

- `alpha = 0.5` → trọng số bằng nhau cho cả 2 nhánh
- `sim_count` = Overlap Coefficient (Count Vectorizer) — phát hiện tên trùng
- `sim_tfidf` = Cosine Similarity (TF-IDF multi-field) — phân biệt ngữ nghĩa

**Dẫn chứng code** (`vectorizer.py`, dòng 248):
```python
final_sim = alpha * sim_count + (1.0 - alpha) * sim_tfidf
```

---

### Q10: Tại sao cần cả 2 nhánh? Không dùng 1 nhánh đủ không?

**Trả lời:**

Mỗi nhánh có điểm mạnh riêng, kết hợp để bù đắp cho nhau:

| Nhánh | Điểm mạnh | Điểm yếu |
|-------|-----------|----------|
| **Count (Overlap)** | Bắt từ trùng tuyệt đối, không bị ảnh hưởng bởi độ dài câu | Không phân biệt từ quan trọng vs từ phổ biến |
| **TF-IDF (Cosine)** | Phân hóa từ khóa chính/phụ nhờ IDF | Bị ảnh hưởng bởi độ dài câu |

**Ví dụ cụ thể:**
- "Sữa tươi Vinamilk" vs "Sữa tươi TH": Count = 1.0 (giống nhau), TF-IDF < 1.0 (vì "Vinamilk" ≠ "TH")
- "Bánh mì tươi" vs "Bánh mì sandwich": Count = 0.5 (1 từ trùng / 2 từ), TF-IDF có thể cao hơn vì "bánh mì" là từ quan trọng

Ensemble giúp tận dụng cả 2: Count bắt substitute tuyệt đối, TF-IDF phân hóa tinh tế.

---

## PHẦN 4 — Quá Trình Lọc

### Q11: Ngưỡng `threshold = 0.25` hoạt động như thế nào?

**Trả lời:**

```
final_sim(A, B) >= 0.25 → B là substitute của A → LOẠI BỎ
final_sim(A, B) < 0.25  → B là complementary   → GIỮ LẠI
```

Logic lọc trong code:
```python
mask = similarities < threshold  # True = giữ lại
```

Chọn 0.25 vì:
- Quá thấp (0.1): loại nhiều quá, kể cả complementary tốt
- Quá cao (0.5): không lọc được substitute
- 0.25 là ngưỡng thực nghiệm để phân biệt "cùng loại" vs "khác loại"

**Dẫn chứng code** (`config.py`, dòng 93):
```python
ENS_CB_THRESHOLD = 0.25
```

**Dẫn chứng code** (`cb_filter.py`, dòng 187):
```python
mask = similarities < threshold
```

---

### Q12: CB Filter được tính on-demand. Nghĩa là gì? Tại sao không pre-compute toàn bộ?

**Trả lời:**

**Pre-compute** tất cả cặp: ma trận `(36,181 × 36,181)` = **1.3 tỷ cặp** → không thể lưu trong RAM.

**On-demand**: chỉ tính similarity cho các cặp `(sản phẩm đầu vào, các candidates)` tại thời điểm gọi:

**Dẫn chứng code** (`cb_filter.py`, dòng 179–182):
```python
similarities = cb_ensemble_similarity(
    self.product_vectors_tfidf, self.product_vectors_count,
    idx_a, valid_indices, alpha=self.alpha,
)
mask = similarities < threshold
```

Mỗi lần recommend chỉ có `top_k = 100` candidates → tính 100 similarity thay vì 36K × 36K.

---

### Q13: Nếu sản phẩm không có vector (cold-start), CB Filter xử lý thế nào?

**Trả lời:**

**Sản phẩm đầu vào không có vector**: trả về toàn bộ candidates không lọc:
```python
if product_a_id not in self.product_id_to_idx:
    return candidates  # Giữ nguyên
```

**Candidate không có vector**: giữ lại candidate đó (không loại bỏ khi không có thông tin):
```python
if cid not in valid_set:
    result.append((cid, score))  # cold-start: giữ lại
```

→ Nguyên tắc **"khi nghi ngờ, giữ lại"** — tránh loại bỏ sản phẩm hợp lệ vì thiếu thông tin.

**Dẫn chứng code** (`cb_filter.py`, dòng 153–155 và 196–197):
```python
if product_a_id not in self.product_id_to_idx:
    return candidates

if cid not in valid_set:
    result.append((cid, score))      # cold-start: giữ lại
```

---

### Q14: Overlap Coefficient xử lý trường hợp mẫu số = 0 thế nào?

**Trả lời:**

Đây là lỗi chia cho 0 (division by zero). Code xử lý bằng `np.divide` với tham số `where`:

**Dẫn chứng code** (`vectorizer.py`, dòng 243–245):
```python
sim_count = np.divide(intersection, min_lengths,
                      out=np.zeros_like(intersection, dtype=float),
                      where=min_lengths != 0)
```

- `out=np.zeros_like(...)`: Nếu mẫu số = 0, gán kết quả = 0
- `where=min_lengths != 0`: Chỉ tính chia khi mẫu số khác 0

---

## PHẦN 5 — So Sánh & Đánh Giá

### Q15: So sánh CB và Item-CF?

**Trả lời:**

| Tiêu chí | CB (Content-Based) | Item-CF (Collaborative) |
|----------|-------------------|------------------------|
| **Dữ liệu đầu vào** | Tên, aisle, department | Lịch sử mua hàng (co-occurrence) |
| **Cách hoạt động** | Tính similarity nội dung | Đếm đồng xuất hiện |
| **Vai trò** | Bộ lọc hậu xử lý | Model gợi ý chính |
| **Phát hiện substitute** | ✅ Tốt | ❌ Không |
| **Phát hiện complementary** | ❌ Không | ✅ Tốt |
| **Cold-start** | ✅ Xử lý được | ❌ Không |
| **Cần dữ liệu mua hàng** | ❌ Không cần | ✅ Cần |
| **Hướng gợi ý** | ❌ Đối xứng | ✅ Có hướng |

---

### Q16: Tại sao CB không tham gia ensemble?

**Trả lời:**

CB không tham gia ensemble vì:

1. **Bản chất khác:** Ensemble kết hợp các model co-occurrence để tính weighted score cho gợi ý. CB không phải model gợi ý, nó là bộ lọc.

2. **Không có score gợi ý:** CB chỉ trả về similarity (0-1), không phải score gợi ý có hướng. Không thể kết hợp với Item-CF score.

3. **Vai trò riêng:** CB làm nhiệm vụ loại bỏ substitute — việc này nên làm sau cùng, không nên pha trộn vào quá trình tính score.

---

### Q17: Hạn chế lớn nhất của CB là gì?

**Trả lời:**

**Hạn chế lớn nhất: Similarity = 0 không có thông tin.**

Khi 2 sản phẩm có tên hoàn toàn khác nhau (VD: "Bia" và "Snack"), similarity = 0. CB không thể kết luận:
- Đây là complementary (nên gợi ý)?
- Hay chỉ đơn giản là không biết?

Vì similarity = 0 < threshold (0.25) → CB giữ lại. Nhưng việc giữ lại này không dựa trên bằng chứng nào, chỉ là "không biết nên không loại".

**Hạn chế khác:**
- Chỉ dựa trên thông tin mô tả, không capture hành vi mua hàng thực tế
- Không thể gợi ý chủ động, chỉ làm nhiệm vụ lọc

---

### Q18: Khi nào CB filter gây hại thay vì có lợi?

**Trả lời:**

**Có.** Các tình huống CB có thể gây hại:

1. **Threshold quá thấp:** Nếu threshold = 0.1, CB loại bỏ quá nhiều → giảm số lượng gợi ý, có thể loại nhầm complementary

2. **Sản phẩm cùng aisle nhưng complementary:** VD: "Bánh mì" và "Bơ" cùng aisle "Bánh mì & Bơ" → similarity cao → có thể bị loại nhầm dù thực tế là complementary

3. **Tên sản phẩm quá giống nhưng khác loại:** VD: "Sữa tươi" và "Sữa chua" có similarity cao nhưng là complementary (sữa chua ăn kèm bánh mì)

---

### Q19: Nếu được cải thiện CB, em sẽ cải thiện thế nào?

**Trả lời:**

3 hướng cải thiện:

1. **Tích hợp CB ngay trong công thức score của Item-CF:** Thay vì làm hậu xử lý, thêm thành phần phạt similarity nội dung vào score:
   ```
   new_score = old_score × (1 - CB_similarity)
   ```
   Nếu A và B quá giống nhau → score bị giảm → ít có cơ hội lọt vào top-K.

2. **Dùng embedding thay vì TF-IDF:** Dùng PhoBERT hoặc các pre-trained Vietnamese embedding để vector hóa tên sản phẩm — capture ngữ nghĩa tốt hơn TF-IDF.

3. **Threshold động:** Thay vì threshold cố định 0.25, dùng threshold phụ thuộc vào aisle:
   - Cùng aisle → threshold cao hơn (dễ bị loại hơn)
   - Khác aisle → threshold thấp hơn (khó bị loại hơn)

---

## TÓM TẮT LUỒNG THUẬT TOÁN

```
products.parquet (tên tiếng Việt: product_name, aisle, department)
        ↓
  Tiền xử lý: lowercase → xóa đơn vị đo → xóa stopwords → xóa ký tự đặc biệt
        ↓ (vectorizer.py dòng 78–99)
  Nhánh 1 — TF-IDF multi-field:
    TF-IDF(name, ngram(1,2), 15000 features) × 1.0
    TF-IDF(aisle, ngram(1,1), 500 features) × 0.8
    TF-IDF(dept, ngram(1,1), 100 features) × 0.6
    → hstack → ma trận (36181 × 15600)
        ↓ (vectorizer.py dòng 199–211)
  Nhánh 2 — Count Vectorizer:
    Count(name, ngram(1,1), 15000 features)
    → ma trận (36181 × 15000)
        ↓ (vectorizer.py dòng 142–153)
  Lưu cả 2 ma trận sparse (cb_tfidf_vectors.npz, cb_count_vectors.npz)
        ↓  khi lọc:
  Lấy vector A và vectors candidates
  final_sim = 0.5 × overlap_coefficient + 0.5 × cosine_tfidf
  Loại candidates có final_sim >= 0.25 (substitute)
  Giữ candidates có final_sim < 0.25 (complementary)
        ↓ (cb_filter.py dòng 179–187)