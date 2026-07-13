# Bộ Câu Hỏi & Trả Lời — Ensemble Model & Đánh Giá
## Weighted Ensemble + CB Filter + LLM Evaluation

---

## 📖 Thuật ngữ tiếng Anh chuyên ngành (Glossary)

| # | Thuật ngữ | Phiên âm (IPA) | Giải thích | Liên hệ dự án |
|---|-----------|---------------|------------|---------------|
| 1 | **Ensemble** | /ɒnˈsɒm.bəl/ | Kết hợp nhiều model để tận dụng điểm mạnh từng model | `0.5×ItemCF + 0.25×I2V + 0.25×KG` (`config.py` dòng 88–90) |
| 2 | **Precision@K** | /prɪˈsɪʒ.ən æt keɪ/ | Số gợi ý đúng trong top-K / K | Precision@10: trong 10 gợi ý, bao nhiêu cái đúng? |
| 3 | **Recall@K** | /rɪˈkɔːl æt keɪ/ | Số gợi ý đúng trong top-K / tổng số complementary | Recall@10: gợi ý được bao nhiêu %? |
| 4 | **F1@K** | /ef wʌn æt keɪ/ | Trung bình điều hòa giữa Precision và Recall | F1 = 2×P×R/(P+R) |
| 5 | **Hit@K** | /hɪt æt keɪ/ | 1 nếu có ít nhất 1 gợi ý đúng trong top-K | Hit@10: có ít nhất 1 gợi ý hữu ích? |
| 6 | **Ground Truth** | /ɡraʊnd truːθ/ | Nhãn thực tế để đánh giá model | Dùng Gemini 2.0 Flash làm ground truth |
| 7 | **LLM Evaluation** | /el el em ɪˌvæl.juˈeɪ.ʃən/ | Dùng Large Language Model làm người đánh giá | Gemini 2.0 Flash đánh giá complementary (`scripts/model/07_eval_llm.py`) |
| 8 | **Min-Max Normalization** | /mɪn mæks ˌnɔː.mə.laɪˈzeɪ.ʃən/ | Chuẩn hóa về [0,1]: (x - min) / (max - min) | `_normalize()` (`ensemble.py` dòng 59–76) |
| 9 | **CB Filter** | /siː biː ˈfɪl.tər/ | Content-Based Diversity Filter, loại substitute | `ENS_CB_THRESHOLD = 0.25` (`config.py` dòng 93) |
| 10 | **Fair Benchmark** | /feər ˈbenʧ.mɑːk/ | Đánh giá công bằng: chỉ so sánh trên tập giao (intersection) của tất cả model | `common_product_ids = set(raw_gt.keys()) & item_cf_vocab & i2v_vocab & kg_vocab & ensemble_vocab` (`07_eval_llm.py`) |

---

## PHẦN 1 — Kiến Trúc Ensemble

### Q1: Ensemble model kết hợp các sub-model như thế nào? Công thức tổng quát là gì?

**Trả lời:**

```
final_score(A → B) = α × ItemCF_score_norm(A, B)
                    + β × Item2Vec_sim_norm(A, B)
                    + γ × KGMetapath_sim_norm(A, B)
```

Với `α = 0.5, β = 0.25, γ = 0.25` (tổng = 1.0).

**Trước khi cộng, mỗi score được normalize** về [0,1] bằng Min-Max để các sub-model có cùng scale (Item-CF score và cosine similarity có phạm vi giá trị rất khác nhau).

Lý do chọn `α = 0.5` cho Item-CF cao hơn:
- Item-CF (Ochiai + Confidence) trực tiếp đo **xác suất mua kèm** → đáng tin nhất
- Item2Vec và KGMetapath học embedding → gián tiếp hơn, trọng số thấp hơn

**Dẫn chứng code** (`config.py`, dòng 88–90):
```python
ENS_ALPHA = 0.5    # Item-CF
ENS_BETA = 0.25   # Item2Vec
ENS_GAMMA = 0.25   # KGMetapath
```

---

### Q2: Tại sao phải normalize score trước khi cộng?

**Trả lời:**

3 sub-model có **phạm vi score khác nhau**:
- Item-CF: `ochiai × conf × log1p(cnt)` — thường trong khoảng [0, 0.1]
- Item2Vec: cosine similarity — trong khoảng [0.3, 0.99] (các embedding gần nhau)
- KGMetapath: cosine similarity — tương tự Item2Vec

Nếu cộng trực tiếp, Item2Vec và KGMetapath sẽ **áp đảo** Item-CF chỉ vì phạm vi giá trị lớn hơn, dù α lớn hơn.

**Min-Max Normalization:**
```
score_norm = (score - min) / (max - min + 1e-9)
```

**Dẫn chứng code** (`ensemble.py`, dòng 59–76):
```python
def _normalize(self, scores):
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)  # Tất cả bằng nhau → 0.5
    return [(s - min_s) / (max_s - min_s + 1e-9) for s in scores]
```

---

### Q3: Luồng đầy đủ của hàm `recommend()` gồm những bước nào?

**Trả lời:**

**6 bước:**

1. **Lấy top-K candidates từ mỗi sub-model** (top_k = 100 candidates/model):
```python
item_cf_recs = self.item_cf.recommend(product_id, top_k=100)
i2v_recs     = self.item2vec.recommend(product_id, top_k=100)
mw_recs      = self.metapath2vec.recommend(product_id, top_k=100)
```

2. **Union** 3 tập candidates → có thể lên đến 300 unique candidates

3. **Tính weighted score** cho mỗi candidate:
```python
final_score[B] = α × norm(itemcf[B]) + β × norm(i2v[B]) + γ × norm(mw[B])
# Nếu sub-model không có B → score = 0
```

4. **Sort giảm dần** theo final_score

5. **CB Filter** loại substitute (nếu `use_cb_filter=True`)

6. **Lấy top final_k** (= 10) kết quả

**Dẫn chứng code** (`ensemble.py`, dòng 152–189):
```python
def recommend(self, product_id: int, use_cb_filter: bool = True, top_k: int = None):
    if top_k is None:
        top_k = self.final_k
    
    candidates_sorted = self.get_raw_candidates(product_id, top_k=self.top_k)
    
    if not candidates_sorted:
        return []
    
    # CB Filter (optional)
    if use_cb_filter and self.cb_filter is not None:
        candidates_sorted = self.cb_filter.filter(
            product_id, candidates_sorted, threshold=ENS_CB_THRESHOLD
        )
    
    # Lấy top_k
    result = candidates_sorted[:top_k]
    return result
```

---

### Q4: Nếu một sản phẩm chỉ có trong 2/3 sub-model thì xử lý thế nào?

**Trả lời:**

Sản phẩm vẫn được tính — missing score được thay bằng **0**:

**Dẫn chứng code** (`ensemble.py`, dòng 128–131):
```python
item_cf_scores = [item_cf_dict.get(pid, 0) for pid in candidate_list]
i2v_scores     = [i2v_dict.get(pid, 0) for pid in candidate_list]
mw_scores      = [mw_dict.get(pid, 0) for pid in candidate_list]
```

Sau normalize, `0` sẽ thành `0` trong normalized scores (min của tập). Sản phẩm vẫn có cơ hội được chọn nếu 2 sub-model kia cho score cao.

**Ý nghĩa**: Union candidates tận dụng được thế mạnh riêng của mỗi model — Item-CF tốt với sản phẩm phổ biến, KGMetapath tốt với sản phẩm ít xuất hiện nhưng cùng danh mục.

---

## PHẦN 2 — Dữ Liệu Đánh Giá (Ground Truth)

### Q5: Ground truth dùng để đánh giá được tạo ra từ đâu? Cấu trúc như thế nào?

**Trả lời:**

Ground truth được sinh bởi **LLM (Gemini)** qua khảo sát — mỗi dòng là 1 cặp sản phẩm mua kèm hợp lý:

```csv
product_A_id,product_B_id,description
1,196,Bữa tiệc ngọt và nước giải khát
1,27125,Bữa tiệc ngọt và nước giải khát
4,24020,Mi y kem ca phe
```

- **`product_A_id`**: Sản phẩm "anchor" — sản phẩm đầu vào cần gợi ý
- **`product_B_id`**: Sản phẩm "ground truth" — sản phẩm nên được gợi ý kèm
- **`description`**: Bối cảnh/lý do mua kèm (dùng để tham khảo, không dùng trong metric)

1 product_A có thể có nhiều product_B → `gt[product_A] = {product_B1, product_B2, ...}` (set).

---

### Q6: Tại sao dùng LLM để tạo ground truth thay vì dùng test set của Instacart?

**Trả lời:**

**Test set Instacart không phù hợp** vì:
1. `eval_set='test'` là **đơn hàng kế tiếp** của user — là bài toán **next-basket prediction**, không phải mua kèm (bundle) trong cùng 1 đơn
2. Sản phẩm trong đơn tiếp theo có thể là reorder sản phẩm cũ, không liên quan đến "mua kèm"

**LLM (Gemini) đóng vai trò "human annotator" tự động:**
- LLM được cung cấp tên sản phẩm A và danh sách candidates
- LLM đánh giá: "B có hợp lý để mua kèm với A không?"
- Output: danh sách (A, B) hợp lệ + lý do (description)

Đây là cách phổ biến trong các hệ thống gợi ý khi không có explicit ground truth cho bài toán bundle.

---

## PHẦN 3 — Metrics Đánh Giá

### Q7: Em đánh giá bằng những metrics nào? Giải thích từng metric. Lý do chọn 4 metrics này?

**Trả lời:**

4 metrics, đều tính **@10** (top-10 recommendations). Trước hết cần hiểu 3 khái niệm cơ bản:

| Khái niệm | Ý nghĩa | Trong dự án |
|-----------|---------|-------------|
| **TP (True Positive)** | Gợi ý **đúng** — sản phẩm được gợi ý VÀ là complementary thật | Model gợi ý B, LLM xác nhận B mua kèm được với A |
| **FP (False Positive)** | Gợi ý **sai** — sản phẩm được gợi ý nhưng KHÔNG phải complementary | Model gợi ý B, LLM nói B không mua kèm với A |
| **FN (False Negative)** | **Bỏ sót** — sản phẩm là complementary thật nhưng model KHÔNG gợi ý | LLM có B trong ground truth, nhưng model không gợi ý B |

---

**1. Precision@10:**
```
P@10 = TP / (TP + FP) = (số ground truth items trong top-10) / 10
```
→ **Trong 10 gợi ý, bao nhiêu cái thực sự là "mua kèm tốt"?**

Khi FP cao (gợi ý sai nhiều) → Precision giảm. CB Filter giúp giảm FP bằng cách loại substitute → tăng Precision.

**Ví dụ:** Model gợi ý 10 sản phẩm cho A, trong đó 6 sản phẩm thực sự là complementary (TP=6), 4 sản phẩm không phải (FP=4) → P@10 = 6/10 = 0.6.

**2. Recall@10:**
```
R@10 = TP / (TP + FN) = (số ground truth items trong top-10) / (tổng ground truth items của A)
```
→ **Trong tất cả "mua kèm tốt" của A, model tìm được bao nhiêu %?**

Khi FN cao (bỏ sót nhiều) → Recall giảm.

**Ví dụ:** A có 20 ground truth items (LLM xác nhận), model gợi ý được 6 cái trong top-10 → R@10 = 6/20 = 0.3.

**3. F1@10:**
```
F1@10 = 2 × P@10 × R@10 / (P@10 + R@10 + 1e-10)
```
→ **Harmonic mean của Precision và Recall** — 1 con số tổng hợp.

Precision và Recall thường **trade-off**: Precision cao thì Recall thường thấp và ngược lại.
- Model A: P=0.8, R=0.2 → F1 = 2×0.8×0.2/(1.0) = **0.32**
- Model B: P=0.5, R=0.5 → F1 = 2×0.5×0.5/(1.0) = **0.50**
Model B tốt hơn dù Precision thấp hơn, vì cân bằng hơn.

→ **Lý do cần F1:** Không thể chỉ dùng Precision hay Recall riêng lẻ.

**4. Hit@10:**
```
Hit@10 = 1 nếu TP ≥ 1 (có ít nhất 1 ground truth trong top-10), ngược lại = 0
```
→ **Có gợi ý được ít nhất 1 sản phẩm hợp lý không?**

**Khác biệt với Precision:** 
- P@10 = 0.1: trong 10 gợi ý, chỉ 1 đúng (vẫn biết chính xác tỷ lệ)
- Hit@10 = 1: có ít nhất 1 đúng (không biết tỷ lệ, chỉ biết có hay không)

Hit@10 đo "có gợi ý được gì hữu ích không?" — quan trọng vì người dùng chỉ cần 1 gợi ý tốt là đã hài lòng.

---

**Tại sao chọn 4 metrics này?** Vì không metric nào hoàn hảo:
- Chỉ Precision → model chỉ gợi ý 1 sản phẩm chắc chắn đúng, P=1.0 nhưng vô dụng
- Chỉ Recall → model gợi ý tất cả, R=1.0 nhưng toàn rác
- F1 cân bằng, Hit đo "có ích hay không"
- Báo cáo cả 4 → nhìn toàn diện

---

### Q8: Tại sao mẫu số của Precision là 10 (cố định) thay vì `len(pred_ids)` thực tế?

**Trả lời:**

Vì em cố định `top_k = 10` — tức là **luôn kỳ vọng model đưa ra đúng 10 gợi ý**. Nếu model trả về ít hơn 10 (ví dụ sản phẩm cold-start chỉ ra được 3), những slot trống bị tính là sai.

Điều này tạo ra **đánh giá công bằng và nghiêm khắc**: model phải chịu phạt khi không đưa đủ 10 gợi ý. Nếu dùng `len(pred_ids)` làm mẫu số, model cold-start chỉ ra 1 gợi ý đúng sẽ có Precision = 1.0 — không phản ánh thực tế.

---

### Q9: "Fair Benchmark" là gì? Tại sao cần lấy tập giao (intersection) của 4 model?

**Trả lời:**

**Vấn đề**: Mỗi model có vocabulary khác nhau:
- Item-CF: chỉ có sản phẩm xuất hiện ≥ 10 lần và có co-occurrence
- Item2Vec: chỉ có sản phẩm xuất hiện ≥ `min_count = 10`
- KGMetapath: tương tự + cần có edge trong đồ thị
- Ground truth: bất kỳ sản phẩm nào

Nếu đánh giá trên tập ground truth đầy đủ:
- Sản phẩm A không có trong Item-CF → trả về `[]` → Precision = 0
- Nhưng Item2Vec có thể biết A → trả về kết quả → Precision > 0

Đây là đánh giá **không công bằng** — Item-CF bị phạt vì cold-start dù thực ra không phải model kém.

**Giải pháp — Intersection:**
```python
common_product_ids = (
    set(raw_gt.keys()) & 
    item_cf_vocab & i2v_vocab & kg_vocab & ensemble_vocab
)
filtered_gt = {pid: raw_gt[pid] for pid in common_product_ids}
```
→ Chỉ đánh giá trên những sản phẩm mà **tất cả 4 model đều biết** → so sánh công bằng.

---

### Q10: Nếu model crash hoặc exception khi gợi ý cho 1 sản phẩm, xử lý thế nào?

**Trả lời:**

Exception được **catch và ghi nhận** — không crash toàn bộ evaluation:

```python
try:
    recs = model.recommend(pid_a, top_k=top_k)
    if recs:
        pred_ids = [pid for pid, _ in recs[:n_recs]]
except Exception as e:
    n_errors += 1
    # pred_ids giữ nguyên = []
```

Lượt đó `pred_ids = []` → `n_correct = 0` → Precision = Recall = F1 = Hit = 0.

Điều này vẫn **công bằng**: nếu model thực sự có vấn đề với sản phẩm đó, điểm 0 là hợp lý. Tổng số lỗi (`n_errors`) được báo cáo trong bảng kết quả để có thể kiểm tra.

---

## PHẦN 4 — Kết Quả & Nhận Định

### Q11: Trong bảng kết quả, em kỳ vọng model nào tốt nhất? Tại sao?

**Trả lời:**

**Kỳ vọng: Ensemble + CB tốt nhất**, vì:

1. **Ensemble (w/o CB)** > Individual models: kết hợp 3 góc nhìn khác nhau —
   - Item-CF: co-occurrence trực tiếp từ dữ liệu mua hàng
   - Item2Vec: embedding ngữ cảnh order
   - KGMetapath: embedding có cấu trúc danh mục

2. **Ensemble + CB** > Ensemble w/o CB: CB Filter loại substitute → kết quả đa dạng hơn, precision cao hơn

3. **Item-CF** thường mạnh baseline vì công thức Ochiai × Confidence × Log trực tiếp đo xác suất mua kèm

4. **KGMetapath** tốt hơn Item2Vec với sản phẩm ít xuất hiện (semantic walk qua aisle bổ sung thông tin danh mục)

**Về Recall**: Ensemble luôn có Recall cao hơn vì union candidates từ 3 model → coverage rộng hơn bất kỳ model đơn lẻ nào.

---

### Q12: Precision@10 và Recall@10 của em có cùng chiều tốt không? Có thể xảy ra trường hợp nào không như kỳ vọng?

**Trả lời:**

Thường có **trade-off**: Precision cao thì Recall có thể thấp và ngược lại.

Trường hợp đặc biệt có thể xảy ra:

**CB Filter có thể giảm Recall**: khi CB Filter loại bỏ 1 candidate, nếu candidate đó lại là ground truth → Recall giảm. Đây là chi phí của diversity.

**Giải thích lý thuyết:**
- Ground truth (LLM) gồm cả sản phẩm complementary lẫn đôi khi sản phẩm tương tự (vd: 2 loại sữa cùng nhóm bữa sáng)
- CB Filter loại sản phẩm tương tự → có thể loại ground truth
- → CB Filter tăng Precision (đa dạng hơn) nhưng có thể giảm Recall

Đây là **lý do chính để báo cáo cả P, R, F1, và Hit** — không chỉ 1 metric.

---

### Q13: LLM Evaluation có hạn chế gì?

**Trả lời:**

**Có. 3 hạn chế chính:**

1. **LLM bias:** Gemini có thể thiên vị sản phẩm phổ biến, không biết sản phẩm đặc thù vùng miền

2. **Chi phí API:** Gọi Gemini API cho hàng nghìn mẫu → tốn kém

3. **Thiếu ngữ cảnh thực tế:** LLM không phải người dùng thật, không biết thói quen mua sắm thực tế

**Giải pháp trong dự án:**
- Dùng prompt chi tiết, cung cấp tên sản phẩm + aisle + department
- Lấy 50-50 top-5 và noise để giảm bias
- So sánh tương đối giữa các model (cùng bộ mẫu) → bias ảnh hưởng như nhau

---

### Q14: Làm thế nào để cải thiện đánh giá?

**Trả lời:**

3 hướng cải thiện:

1. **Human Evaluation:** Thay vì chỉ dùng LLM, thuê người thật đánh giá (A/B testing) — chính xác nhất nhưng tốn kém

2. **Cross-validation:** Thay vì 1 lần đánh giá, chia survey samples thành nhiều fold, đánh giá nhiều lần để giảm variance

3. **Thêm metrics:** 
   - **Mean Reciprocal Rank (MRR):** Vị trí của gợi ý đúng đầu tiên
   - **NDCG@K (Normalized Discounted Cumulative Gain):** Đánh giá thứ tự gợi ý
   - **Coverage:** Tỷ lệ sản phẩm được gợi ý (đo độ phủ)

---

## TÓM TẮT LUỒNG ENSEMBLE + ĐÁNH GIÁ

```
                    ┌─────────────────────────────────────────┐
                    │          INFERENCE (recommend)           │
                    └─────────────────────────────────────────┘
product_id_A
     │
     ├──► Item-CF.recommend(A, top_k=100) → [(B1, s1), (B2, s2), ...]
     ├──► Item2Vec.recommend(A, top_k=100) → [(B3, s3), ...]
     └──► KGMetapath.recommend(A, top_k=100) → [(B5, s5), ...]
     │                  (ensemble.py dòng 106–108)
     │
     Union candidates (up to 300 unique)
     │                  (ensemble.py dòng 111–117)
     │
     Min-Max Normalize mỗi sub-model scores
     │                  (ensemble.py dòng 59–76)
     │
     Weighted sum: 0.5×CF + 0.25×I2V + 0.25×KG
     │                  (ensemble.py dòng 137–140)
     │
     Sort giảm dần
     │
     CB Filter (threshold=0.25): loại substitute
     │                  (ensemble.py dòng 181–184)
     │
     Top-10 kết quả
     │
     ▼
                ┌──────────────────────────────────────┐
                │           EVALUATION                  │
                └──────────────────────────────────────┘
                    │
              ground_truth = load LLM CSV
              filtered_gt = ground_truth ∩ common_vocab(4 models)
                    │
              For each product_A in filtered_gt:
                pred_ids = model.recommend(A, top_k=10)
                n_correct = |pred_ids ∩ gt[A]|
                P@10 = n_correct / 10
                R@10 = n_correct / |gt[A]|
                F1@10 = harmonic_mean(P, R)
                Hit@10 = 1 if n_correct > 0
                    │
              Mean over all products → báo cáo kết quả