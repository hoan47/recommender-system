# Bộ Câu Hỏi & Trả Lời — Item-Based Collaborative Filtering (Item-CF)
## Ochiai Coefficient + Confidence Score + Log Frequency

---

## 📖 Thuật ngữ tiếng Anh chuyên ngành (Glossary)

> Các thuật ngữ này giáo viên có thể hỏi bất ngờ. Mỗi thuật ngữ đều có dẫn chứng code/file trong dự án.

| # | Thuật ngữ | Phiên âm (IPA) | Giải thích | Liên hệ dự án |
|---|-----------|---------------|------------|---------------|
| 1 | **Collaborative Filtering (CF)** | /kəˈlæb.ər.ə.tɪv ˈfɪl.tər.ɪŋ/ | Lọc cộng tác: gợi ý dựa trên hành vi của nhiều người dùng, không cần thông tin mô tả sản phẩm | Item-CF là một nhánh của CF, dựa trên sản phẩm (Item-Based) |
| 2 | **Memory-based** | /ˈmem.ər.i beɪst/ | Phương pháp "ghi nhớ": lưu trữ dữ liệu và tra cứu trực tiếp, không học tham số | `ItemCFModel.fit()` chỉ đếm co-occurrence, không tối ưu loss (`item_cf.py` dòng 44) |
| 3 | **Co-occurrence** | /ˌkəʊ.əˈkʌr.əns/ | Đồng xuất hiện: hai sản phẩm cùng xuất hiện trong cùng một đơn hàng | `count_pairs_numba()` đếm co-occurrence pairs (`_numba_ops.py`) |
| 4 | **CSR Matrix** | /siː.esˈɑːr ˈmeɪ.trɪks/ | Định dạng ma trận thưa, lưu 3 mảng: `data`, `indices`, `indptr` | `sparse.csr_matrix(...)` `item_cf.py` dòng 123–127 |
| 5 | **Ochiai Coefficient** | /əʊˈtʃaɪ.iː ˌkəʊ.ɪˈfɪʃ.ənt/ | `cnt / sqrt(cnt_A × cnt_B)` — normalize co-occurrence theo độ phổ biến | Tính tại `item_cf.py` dòng 164–173 |
| 6 | **Confidence Score** | /ˈkɒn.fɪ.dəns skɔːr/ | Xác suất có điều kiện P(B\|A) = cnt(A,B) / count(A) — tạo tính bất đối xứng | `conf = row / cnt_i` `item_cf.py` dòng 176 |
| 7 | **Complementary** | /ˌkɒm.plɪˈmen.tər.i/ | Mua kèm: sản phẩm bổ sung cho nhau (bia + snack) | Mục tiêu của dự án: gợi ý complementary |
| 8 | **Substitute** | /ˈsʌb.stɪ.tjuːt/ | Thay thế: sản phẩm có thể thay thế nhau (Coca + Pepsi) | Item-CF không phân biệt, CB Filter loại bỏ (threshold=0.25) |
| 9 | **Cold-start** | /kəʊld stɑːt/ | Sản phẩm mới chưa có dữ liệu lịch sử mua hàng | `recommend()` trả về `[]` nếu không có mapping (`item_cf.py` dòng 199–200) |
| 10 | **Long-tail** | /lɒŋ teɪl/ | Sản phẩm hiếm, ít xuất hiện (46.3% sản phẩm có < 50 lần tương tác) | Item-CF yếu, KGMetapath xử lý tốt hơn nhờ semantic walk |
| 11 | **Min-support** | /mɪn səˈpɔːt/ | Ngưỡng lọc: cặp xuất hiện dưới ngưỡng sẽ bị loại bỏ | `ITEMCF_MIN_SUPPORT = 10` (`config.py` dòng 54) |
| 12 | **Numba JIT** | /ˈnʌm.bə dʒɪt/ | Just-In-Time compilation: biên dịch Python sang machine code tại runtime | `@njit` trên `count_pairs_numba()` (`_numba_ops.py` dòng 16–17) |
| 13 | **Binary Vector** | /ˈbaɪ.nər.i ˈvek.tər/ | Vector chỉ gồm 0 và 1 | Mỗi sản phẩm là binary vector với chiều là các order |
| 14 | **Precision@K** | /prɪˈsɪʒ.ən æt keɪ/ | Số gợi ý đúng trong top-K / K | Precision@10: trong 10 gợi ý, bao nhiêu cái đúng? |
| 15 | **Recall@K** | /rɪˈkɔːl æt keɪ/ | Số gợi ý đúng trong top-K / tổng số complementary | Recall@10: gợi ý được bao nhiêu %? |
| 16 | **F1@K** | /ef wʌn æt keɪ/ | Trung bình điều hòa giữa Precision và Recall | F1 = 2×P×R/(P+R) |
| 17 | **Hit@K** | /hɪt æt keɪ/ | 1 nếu có ít nhất 1 gợi ý đúng trong top-K, ngược lại 0 | Hit@10: có ít nhất 1 gợi ý hữu ích? |
| 18 | **Overfitting** | /ˌəʊ.vəˈfɪt.ɪŋ/ | Học quá khớp: model nhớ nhiễu thay vì học pattern tổng quát | min_support=10 chống overfitting cho cặp hiếm |
| 19 | **Sparse Matrix** | /spɑːs ˈmeɪ.trɪks/ | Ma trận thưa: hầu hết phần tử bằng 0, chỉ lưu phần tử khác 0 | `cooc_matrix` là CSR sparse (36K×36K nhưng nnz « 36K²) |
| 20 | **Ensemble** | /ɒnˈsɒm.bəl/ | Kết hợp nhiều model để tận dụng điểm mạnh từng model | `0.5×ItemCF + 0.25×I2V + 0.25×KG` (`config.py` dòng 88–90) |

---

## PHẦN 1 — Dữ Liệu Đầu Vào

### Q1: Đầu vào của Item-CF là gì? Tại sao không dùng trực tiếp ratings như collaborative filtering truyền thống?

**Trả lời:**

Đầu vào là `order_products.parquet` — bảng giao dịch gồm các cột `[order_id, product_id, ...]`. Đây là **implicit feedback** (không có điểm đánh giá rõ ràng), chỉ biết sản phẩm nào xuất hiện cùng nhau trong cùng 1 đơn hàng.

Collaborative Filtering truyền thống (user-item matrix với rating 1–5) không áp dụng được vì:
- Instacart không có rating — chỉ có "mua" hoặc "không mua"
- Bài toán là **gợi ý mua kèm** (bundle) chứ không phải gợi ý cho user cụ thể
- Tập trung vào quan hệ **item–item** thay vì user–item

**Dẫn chứng code** (`item_cf.py`, dòng 64–65):
```python
grouped = order_products.groupby('order_id')['product_id']
```
→ Group theo `order_id` để lấy các sản phẩm trong cùng 1 đơn hàng.

---

### Q2: Co-occurrence matrix là gì? Tại sao phải xây nó?

**Trả lời:**

Co-occurrence matrix `cooc_matrix[A][B]` = **số lần sản phẩm A và B xuất hiện cùng nhau trong cùng 1 đơn hàng**.

Ý nghĩa: nếu A và B thường mua cùng nhau, giá trị `cooc[A][B]` cao → có thể gợi ý B khi user mua A.

Ma trận này là **đối xứng** (`cooc[A][B] = cooc[B][A]`) vì cùng xuất hiện trong 1 đơn hàng là quan hệ hai chiều.

**Dẫn chứng code** (`item_cf.py`, dòng 119–127) — xây CSR matrix đối xứng:
```python
# Ma trận đối xứng: thêm cả (b, a)
rows_all = np.concatenate([rows, cols])
cols_all = np.concatenate([cols, rows])
data_all = np.concatenate([counts, counts])

self.cooc_matrix = sparse.csr_matrix(
    (data_all, (rows_all, cols_all)),
    shape=(self.n_products, self.n_products),
    dtype=np.int32
)
```
→ Thêm cả `(B, A)` để đảm bảo tính đối xứng.

---

### Q3: Tại sao dùng Numba để đếm co-occurrence pairs? Không dùng Python thông thường được không?

**Trả lời:**

Dữ liệu có **31.9 triệu giao dịch**, mỗi đơn hàng trung bình 10 sản phẩm → số pairs tổ hợp C(10,2) = 45 pairs/đơn, nhân với ~3.3 triệu đơn hàng = **~150 triệu pairs**. Python vòng lặp `for` sẽ mất hàng giờ.

Numba `@njit` compile hàm Python sang native machine code, loại bỏ overhead của Python interpreter → nhanh hơn 50–100x.

**Dẫn chứng code** (`_numba_ops.py`, dòng 16–17):
```python
from numba import njit

@njit
def count_pairs_numba(order_indices, order_ptr, n_products):
```

Thuật toán 2 phase:
1. **Phase 1**: Đếm tổng số pairs để pre-allocate array
2. **Phase 2**: Ghi pairs vào array, sort theo key, đếm unique

---

### Q4: `min_support = 10` nghĩa là gì và tại sao cần lọc?

**Trả lời:**

`min_support = 10` nghĩa là chỉ giữ lại các cặp (A, B) xuất hiện **ít nhất 10 lần** trong toàn bộ tập train. Những cặp xuất hiện ít hơn bị coi là nhiễu (noise) và bị loại.

Lý do:
- Nếu A và B chỉ mua cùng nhau 1–2 lần, rất có thể là ngẫu nhiên, không phải pattern thực sự
- Giảm bộ nhớ và tăng tốc độ inference
- Khoảng **14.4%** sản phẩm xuất hiện dưới 10 lần trong toàn bộ prior → long-tail vấn đề phổ biến

**Dẫn chứng code** (`item_cf.py`, dòng 108–113):
```python
if self.min_support > 0:
    mask = counts >= self.min_support
    rows = rows[mask]
    cols = cols[mask]
    counts = counts[mask]
```

**Dẫn chứng config** (`config.py`, dòng 54):
```python
ITEMCF_MIN_SUPPORT = 10     # pair xuất hiện < 10 lần → bỏ qua
```

---

## PHẦN 2 — Công Thức Score

### Q5: Công thức score cuối cùng của Item-CF là gì? Giải thích từng thành phần.

**Trả lời:**

```
score(A → B) = ochiai(A, B) × conf(A → B) × log1p(cnt(A, B))
```

**Thành phần 1 — Ochiai Coefficient:**
```
ochiai(A, B) = cnt(A, B) / sqrt(count(A) × count(B))
```
- `cnt(A, B)` = số lần A và B cùng xuất hiện
- `count(A)` = tổng số đơn hàng có sản phẩm A
- Đây chính là **Cosine similarity trên binary vector** (mỗi sản phẩm là vector nhị phân, chiều là các order)
- **Đối xứng**: `ochiai(A,B) = ochiai(B,A)`
- Normalize về [0,1], triệt tiêu bias sản phẩm phổ biến

**Thành phần 2 — Confidence:**
```
conf(A → B) = cnt(A, B) / count(A)
```
- **Bất đối xứng**: `conf(A→B) ≠ conf(B→A)`
- Ý nghĩa: "Khi mua A, xác suất cũng mua B là bao nhiêu?"
- Có hướng → phù hợp bài toán "mua kèm" (biết A, đoán B)

**Thành phần 3 — Log Frequency Bonus:**
```
log_freq = log1p(cnt(A, B))
```
- Thưởng thêm cho cặp xuất hiện nhiều lần
- Dùng `log1p` (= log(1+x)) thay vì `log` để tránh `log(0)` và làm mượt tăng trưởng

**Dẫn chứng code** (`item_cf.py`, dòng 164–182):
```python
# Ochiai = cnt / sqrt(count(A) * count(B))
ochiai = np.zeros(self.n_products)
nonzero_indices = np.where(row > 0)[0]
cnts = row[nonzero_indices]
counts_j = self.product_counts[nonzero_indices]
mask = counts_j > 0
if mask.any():
    valid_indices = nonzero_indices[mask]
    ochiai[valid_indices] = cnts[mask] / np.sqrt(cnt_i * counts_j[mask])

# Confidence: conf(A→B) = cnt / count(A)
conf = row / cnt_i

# Log frequency: log1p(cnt)
log_freq = np.log1p(row)

# Score cuối
scores = ochiai * conf * log_freq
```

---

### Q6: Tại sao Ochiai lại là Cosine similarity trên binary vector?

**Trả lời:**

Mỗi sản phẩm A được biểu diễn bằng **binary vector** kích thước `n_orders`:
- `v_A[i] = 1` nếu đơn hàng thứ i có sản phẩm A, ngược lại = 0

Cosine similarity giữa `v_A` và `v_B`:
```
cosine(v_A, v_B) = (v_A · v_B) / (||v_A|| × ||v_B||)
                 = |A ∩ B| / (sqrt(|A|) × sqrt(|B|))
                 = cnt(A,B) / sqrt(count(A) × count(B))
                 = ochiai(A, B)
```

Với binary vector: `v_A · v_B = |A ∩ B| = cnt(A,B)` và `||v_A|| = sqrt(count(A))`.

→ Đây là lý do Item-CF được gọi là "Cosine similarity trên binary vector".

**Dẫn chứng code** (`item_cf.py`, dòng 25–31 — docstring của class):
```python
"""
Item-Based Collaborative Filtering (Item-CF) — Ochiai + Confidence Score.

score(A → B) = ochiai(A,B) * conf(A→B) * log1p(cnt(A,B))

- ochiai = cnt / sqrt(count(A) * count(B))  — đối xứng, normalize
  (tương đương Cosine similarity trên binary vector)
- conf_{A→B} = cnt / count(A)               — bất đối xứng, có hướng
"""
```

---

### Q7: Tại sao nhân cả 3 thành phần lại mà không dùng chỉ Ochiai?

**Trả lời:**

Chỉ dùng Ochiai có nhược điểm:

| Vấn đề | Ví dụ | Giải pháp |
|--------|-------|-----------|
| Ochiai đối xứng, không phân biệt hướng | `ochiai(muối → tiêu) = ochiai(tiêu → muối)` | Nhân `conf(A→B)` có hướng |
| Ochiai không phân biệt pair xuất hiện 10 lần vs 1000 lần | Cùng ratio nhưng confidence khác nhau | Nhân `log1p(cnt)` |
| Sản phẩm rất phổ biến (bánh mì) có ochiai cao với tất cả | Không hữu ích khi gợi ý | Ochiai đã normalize nhưng conf giúp lọc thêm |

Nhân 3 thành phần tạo **composite score** cân bằng giữa: similarity đối xứng + hướng dự đoán + frequency.

---

### Q8: Tại sao score của Item-CF có hướng? (A→B khác B→A)

**Trả lời:**

Score có hướng vì công thức chứa thành phần **Confidence**:
```
score(A → B) = ochiai(A,B) × conf(A→B) × log1p(cnt)
```
Trong đó `conf(A→B) = cnt(A,B) / count(A)` — đây là xác suất có điều kiện P(B|A).

Vì `count(A)` và `count(B)` khác nhau nên `conf(A→B) ≠ conf(B→A)`.

---

### Ví dụ thực tế chứng minh lý do cần xác định chiều: Gà và Sả

Giả sử hệ thống ghi nhận số lượng đơn hàng thực tế như sau:
*   Tổng số đơn mua **Gà** (`count(Gà)`): **100 đơn**
*   Tổng số đơn mua **Sả** (`count(Sả)`): **500 đơn**
*   Số đơn mua chung cả **Gà và Sả** (`cnt(Gà, Sả)`): **30 đơn**

**Ý nghĩa thực tế:**
- Nếu A là **nước lẩu** (ít phổ biến) và B là **thịt bò** (rất phổ biến):
  - `conf(nước lẩu → thịt bò)` cao: ai mua nước lẩu gần như chắc chắn mua thịt bò
  - `conf(thịt bò → nước lẩu)` thấp: vì phần lớn người mua thịt bò không mua nước lẩu

**Dẫn chứng code** (`item_cf.py`, dòng 176):
```python
# Confidence: conf(A→B) = cnt / count(A)
conf = row / cnt_i
```

---

### Q9: Log frequency giải quyết vấn đề gì? Tại sao dùng log1p thay vì cnt trực tiếp?

**Trả lời:**

Nếu dùng `cnt` trực tiếp:
- Cặp (cơm, gà) xuất hiện 1,000,000 lần → đóng góp 1,000,000
- Cặp (mù tạt, xúc xích) xuất hiện 50 lần → đóng góp 50

→ Cặp phổ biến áp đảo hoàn toàn, cặp hiếm không có cơ hội.

Dùng `log1p(cnt) = log(1 + cnt)`:
- Cặp (cơm, gà): log(1,000,001) ≈ 13.8
- Cặp (mù tạt, xúc xích): log(51) ≈ 3.9

→ Chênh lệch chỉ còn ~3.5 lần thay vì 20,000 lần.

**Tác dụng:** Giúp các cặp ít phổ biến hơn nhưng vẫn có ý nghĩa (ví dụ sản phẩm theo mùa, đặc sản vùng miền) có cơ hội xuất hiện trong top gợi ý.

**Dẫn chứng code** (`item_cf.py`, dòng 179):
```python
# Log frequency: log1p(cnt)
log_freq = np.log1p(row)
```

---

## PHẦN 3 — So Sánh Với Phương Pháp Khác

### Q10: Tại sao không dùng PMI (Pointwise Mutual Information) hay SPPMI?

**Trả lời:**

PMI có công thức: `PMI(A,B) = log(P(A,B) / (P(A) × P(B)))`

| Vấn đề | PMI/SPPMI | Item-CF (Ochiai + Conf) |
|---------|-----------|------------------------|
| Item phổ biến bị underrate (cơm + gà) | ❌ PMI phạt nặng P(A) lớn | ✅ Ochiai normalize cân bằng |
| Pair hiếm 1-2 lần bị overrate | ⚠️ SPPMI giảm nhưng chưa đủ | ✅ log1p suppress + min_support |
| Chỉ cho score đối xứng | ✅/❌ | ✅ Confidence có hướng |

**Cụ thể:** PMI phạt các sản phẩm phổ biến. Ví dụ "cơm" và "gà" đều rất phổ biến → P(cơm) × P(gà) lớn → PMI thấp dù chúng thường đi cùng nhau. Ochiai không gặp vấn đề này vì nó chia cho sqrt, không chia cho tích.

---

### Q11: Item-CF khác Item2Vec thế nào?

**Trả lời:**

| Tiêu chí | Item-CF (Memory-based) | Item2Vec (Neural-based) |
|----------|----------------------|------------------------|
| **Cách học** | Đếm trực tiếp co-occurrence | Học embedding qua Skip-gram |
| **Quan hệ** | Chỉ capture cặp trực tiếp (A-B) | Capture cả quan hệ gián tiếp (A-B-C → A~C) |
| **Hướng** | ✅ Có hướng (A→B ≠ B→A) | ❌ Đối xứng |
| **Tốc độ train** | Nhanh (chỉ đếm) | Chậm (cần epochs) |
| **Giải thích** | Dễ (có thể trace từng cặp) | Khó (embedding là "hộp đen") |
| **Long-tail** | ⚠️ Cần min_support | ⚠️ Cần min_count |

---

### Q12: Item-CF so với KGMetapath?

**Trả lời:**

| Tiêu chí | Item-CF | KGMetapath |
|----------|---------|------------|
| **Loại** | Item-Based CF (Memory) | Graph-Based (KG embedding) |
| **Dữ liệu** | Chỉ co-occurrence | Co-occurrence + Aisle + Department |
| **Quan hệ** | Bậc 1 (trực tiếp) | Bậc cao (metapath walk) |
| **Long-tail** | ⚠️ Không tốt | ✅ Tốt (semantic walk qua aisle) |
| **Cold-start** | ❌ Không gợi ý được | ⚠️ Có thể qua aisle |
| **Giải thích** | ✅ Dễ | ❌ Khó |

**KGMetapath vượt trội khi:**
- Sản phẩm **long-tail** (xuất hiện < 50 lần): Item-CF không đủ dữ liệu, KGMetapath dùng semantic walk qua aisle để kết nối
- Cần **high-order relationships**: A và C không trực tiếp cùng xuất hiện, nhưng qua B (A-B, B-C) → KGMetapath capture được

---

### Q13: Tại sao normalize bằng sqrt(count) mà không phải max(count)?

**Trả lời:**

Các lựa chọn normalize khác nhau cho ý nghĩa khác nhau:

| Công thức | Tên | Ý nghĩa |
|-----------|-----|---------|
| `cnt / sqrt(cnt_A × cnt_B)` | **Ochiai / Cosine** | Tương đương Cosine trên binary vector |
| `cnt / min(cnt_A, cnt_B)` | **Overlap coefficient** | Quan tâm bên ít xuất hiện hơn |
| `cnt / max(cnt_A, cnt_B)` | **Jaccard (gần đúng)** | Phạt nặng nếu 1 bên quá phổ biến |

**Tại sao chọn sqrt (Ochiai):**
- Cosine similarity là metric chuẩn cho vector similarity
- Nó cân bằng giữa việc không phạt item phổ biến quá mức (như Jaccard) và không thiên vị item hiếm quá mức (như Overlap)
- Có nền tảng toán học vững chắc: góc giữa 2 vector trong không gian

**Dẫn chứng code** (`item_cf.py`, dòng 164–173):
```python
# Ochiai = cnt / sqrt(count(A) * count(B))
ochiai = np.zeros(self.n_products)
nonzero_indices = np.where(row > 0)[0]
cnts = row[nonzero_indices]
counts_j = self.product_counts[nonzero_indices]
mask = counts_j > 0
if mask.any():
    valid_indices = nonzero_indices[mask]
    ochiai[valid_indices] = cnts[mask] / np.sqrt(cnt_i * counts_j[mask])
```

---

## PHẦN 4 — Inference (Recommend)

### Q14: Khi gọi `recommend(product_id)`, model làm gì bước nào?

**Trả lời:**

1. **Lookup index**: `idx = product_id_to_idx[product_id]`
2. **Lấy row từ CSR matrix**: `row = cooc_matrix[idx].toarray()` → array (n_products,) chứa co-occurrence counts với mọi sản phẩm
3. **Tính ochiai** vectorized cho tất cả non-zero entries
4. **Tính conf** = `row / cnt_i`
5. **Tính log_freq** = `log1p(row)`
6. **Nhân 3 thành phần** → `scores`
7. **Bỏ chính nó** (`scores[idx] = -1`)
8. **Argsort giảm dần**, lấy top-K

**Dẫn chứng code** (`item_cf.py`, dòng 185–218):
```python
def recommend(self, product_id: int, top_k: int = None):
    if product_id not in self.product_id_to_idx:
        return []
    
    idx = self.product_id_to_idx[product_id]
    scores = self._compute_scores(idx)
    
    # Bỏ qua chính nó
    scores[idx] = -1
    
    # Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    result = [
        (self.idx_to_product_id[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0
    ]
    return result
```

---

### Q15: Nếu một sản phẩm mới chưa có trong training data, model xử lý thế nào?

**Trả lời:**

Model trả về **empty list `[]`** ngay lập tức — đây là vấn đề **cold-start**:

```python
if product_id not in self.product_id_to_idx:
    return []
```

**Dẫn chứng code** (`item_cf.py`, dòng 199–200):
```python
if product_id not in self.product_id_to_idx:
    return []
```

Sản phẩm mới không có trong `product_id_to_idx` (mapping được xây từ training data) → không có co-occurrence với bất kỳ sản phẩm nào → không thể gợi ý.

Giải pháp trong hệ thống này: Ensemble fallback — nếu Item-CF không có kết quả, Item2Vec hoặc KGMetapath vẫn có thể gợi ý dựa trên semantic (tên, aisle, department).

---

### Q16: Score = 0 khi nào? Điều đó có ý nghĩa gì?

**Trả lời:**

Score(A→B) = 0 trong các trường hợp:
1. **cnt(A,B) = 0**: A và B chưa bao giờ xuất hiện cùng nhau → Ochiai = 0, Conf = 0, Log freq = 0
2. **count(A) = 0**: Sản phẩm A chưa từng xuất hiện trong bất kỳ đơn hàng nào (cold-start)
3. **count(B) = 0**: Sản phẩm B chưa từng xuất hiện → Ochiai không tính được (chia cho 0)

**Dẫn chứng code** (`item_cf.py`, dòng 156–158):
```python
cnt_i = self.product_counts[product_idx]
if cnt_i == 0:
    return np.zeros(self.n_products)  # Cold-start → score = 0
```

---

## PHẦN 5 — Lưu / Load Model

### Q17: Model lưu những gì? Tại sao cần lưu cả `product_counts`?

**Trả lời:**

Model lưu 3 thành phần:
1. `cooc_matrix.npz` — CSR sparse matrix (co-occurrence counts)
2. `metadata.json` — mapping, hyperparameters, `product_counts`, `total_orders`
3. `product_counts.csv` — để dễ đọc/kiểm tra bằng mắt

`product_counts` (array đếm số đơn hàng của mỗi sản phẩm) cần lưu vì:
- Dùng tính `ochiai` (mẫu số `sqrt(count(A) × count(B))`)
- Dùng tính `conf` (mẫu số `count(A)`)
- Nếu không lưu phải tính lại từ đầu = đọc lại toàn bộ 31.9M records

**Dẫn chứng code** (`item_cf.py`, dòng 220–251):
```python
def save(self, path: str):
    os.makedirs(path, exist_ok=True)
    sparse.save_npz(os.path.join(path, "cooc_matrix.npz"), self.cooc_matrix)
    metadata = {
        'min_support': int(self.min_support),
        'n_products': int(self.n_products),
        'product_id_to_idx': {int(k): int(v) for k, v in self.product_id_to_idx.items()},
        'idx_to_product_id': {str(int(k)): int(v) for k, v in self.idx_to_product_id.items()},
        'product_counts': [int(c) for c in self.product_counts],
        'total_orders': int(self.total_orders),
    }
    with open(os.path.join(path, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
```

---

## PHẦN 6 — Tối Ưu & Hạn Chế

### Q18: Tại sao phải dùng CSR matrix? Vì sao không dùng ma trận dense?

**Trả lời:**

Với 36,181 sản phẩm, ma trận vuông có kích thước:
- **Dense**: 36,181 × 36,181 × 4 bytes (int32) ≈ **4.9 GB** — không thể lưu trong RAM
- **CSR sparse**: Chỉ lưu các phần tử khác 0. Sau khi lọc min_support=10, số cặp còn lại khoảng vài triệu → chỉ vài trăm MB

CSR (Compressed Sparse Row) lưu 3 mảng:
- `data`: giá trị khác 0
- `indices`: cột tương ứng
- `indptr`: vị trí bắt đầu mỗi hàng

**Dẫn chứng code** (`item_cf.py`, dòng 123–127):
```python
self.cooc_matrix = sparse.csr_matrix(
    (data_all, (rows_all, cols_all)),
    shape=(self.n_products, self.n_products),
    dtype=np.int32
)
```

---

### Q19: Item-CF có thể gợi ý substitute không? Có hại gì?

**Trả lời:**

**Có.** Item-CF chỉ dựa trên đồng xuất hiện, không phân biệt được:
- **Complementary** (mua kèm): bia + snack → nên gợi ý
- **Substitute** (thay thế): Coca + Pepsi → không nên gợi ý (người dùng chỉ mua 1 trong 2)

Cả 2 đều có co-occurrence cao, Item-CF không thể phân biệt.

**Tác hại:** Trải nghiệm người dùng kém: "Tôi đã mua Coca, sao lại gợi ý Pepsi?"

**Giải pháp:** Dùng **CB Filter** (Content-Based Filtering) làm tầng hậu xử lý:
- Tính similarity nội dung giữa sản phẩm A và B dựa trên tên, aisle, department
- Nếu similarity ≥ threshold (0.25) → coi là substitute → **loại bỏ**

**Dẫn chứng code** (`config.py`, dòng 88–93):
```python
ENS_ALPHA = 0.5                 # trọng số Item-CF (Ochiai)
ENS_BETA = 0.25                 # trọng số Item2Vec
ENS_GAMMA = 0.25                # trọng số Metapath2Vec (IKG embedding)
ENS_FINAL_K = 10                # top-K cuối cùng output
ENS_CB_THRESHOLD = 0.25          # ngưỡng CB filter khi dùng hybrid ensemble
```

---

### Q20: Vai trò của Item-CF trong Ensemble?

**Trả lời:**

```
final_score = 0.5 × ItemCF + 0.25 × Item2Vec + 0.25 × KGMetapath
```

Item-CF được đặt trọng số cao nhất (0.5) vì:

1. **Trực tiếp nhất**: Item-CF đếm trực tiếp số lần A và B cùng xuất hiện — đây là định nghĩa chính xác nhất của "mua kèm"
2. **Có hướng**: Item-CF là model duy nhất có score bất đối xứng, giúp ensemble có được tính hướng
3. **Robust**: Với min_support=10 và log frequency, Item-CF ít bị nhiễu hơn
4. **Dễ giải thích**: Có thể trace tại sao B được gợi ý từ A (vì cnt(A,B) cao)

**Dẫn chứng code** (`config.py`, dòng 88–90):
```python
ENS_ALPHA = 0.5                 # trọng số Item-CF (Ochiai)
ENS_BETA = 0.25                 # trọng số Item2Vec
ENS_GAMMA = 0.25                # trọng số Metapath2Vec (IKG embedding)
```

---

### Q21: Hạn chế lớn nhất của Item-CF là gì? Hướng cải thiện?

**Trả lời:**

**Hạn chế lớn nhất: Không phân biệt được complementary và substitute.**

Item-CF chỉ dựa trên đồng xuất hiện (co-occurrence). Nếu A và B thường xuyên xuất hiện cùng nhau, Item-CF cho score cao — bất kể đó là:
- Mua kèm (bia + snack) ✅
- Hay thay thế (Coca + Pepsi) ❌

**Hướng cải thiện:**
1. **Tích hợp CB filter ngay trong công thức score:** Thêm thành phần phạt nếu similarity nội dung quá cao
2. **Session-aware Item-CF:** Chỉ tính co-occurrence trong cùng session, không tính trong toàn bộ lịch sử
3. **Kết hợp temporal weighting:** Cặp xuất hiện gần đây có trọng số cao hơn cặp xuất hiện từ lâu

---

## TÓM TẮT LUỒNG THUẬT TOÁN

```
order_products.parquet
        ↓
  Group by order_id → lấy list product_idx trong mỗi order
        ↓ (item_cf.py dòng 64)
  count_pairs_numba() → đếm co-occurrence pairs (Numba JIT)
        ↓ (_numba_ops.py dòng 16–17)
  Lọc min_support=10 → loại cặp ít xuất hiện
        ↓ (item_cf.py dòng 108–113)
  Xây CSR matrix đối xứng (n_products × n_products)
        ↓ (item_cf.py dòng 119–127)
  Khi recommend:
  Lấy row A từ CSR → tính ochiai × conf × log1p
        ↓ (item_cf.py dòng 164–182)
  Argsort giảm dần → top-K candidates
        ↓ (item_cf.py dòng 209–216)
  Ensemble + CB Filter → kết quả cuối