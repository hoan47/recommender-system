# Bộ Câu Hỏi & Trả Lời — KG Metapath (Knowledge Graph Embedding)
## Heterogeneous KG + Metapath Walk + Skip-gram

---

## 📖 Thuật ngữ tiếng Anh chuyên ngành (Glossary)

| # | Thuật ngữ | Phiên âm (IPA) | Giải thích | Liên hệ dự án |
|---|-----------|---------------|------------|---------------|
| 1 | **Knowledge Graph (KG)** | /ˈnɒl.ɪdʒ ɡræf/ | Đồ thị tri thức: lưu trữ thông tin dạng node và edge | KG gồm 3 loại node: Product, Aisle, Department (`kg_metapath.py` dòng 34–36) |
| 2 | **Heterogeneous** | /ˌhet.ər.əˈdʒiː.ni.əs/ | Không đồng nhất: nhiều loại node và edge khác nhau | KG có 3 loại node (P, A, D) và 3 loại edge (CO_OCCUR, BELONGS_TO, PART_OF) |
| 3 | **Metapath** | /ˈmet.ə.pæθ/ | Lộ trình định hướng ngữ nghĩa trong heterogeneous graph | 2 kịch bản: Behavioral (P→P→P) và Semantic (P→A→P) |
| 4 | **Random Walk** | /ˈræn.dəm wɔːk/ | Duyệt đồ thị ngẫu nhiên | Metapath Walk là random walk có định hướng (`kg_metapath.py` dòng 221–286) |
| 5 | **Behavioral Walk** | /bɪˈheɪ.vjə.rəl wɔːk/ | Walk theo cạnh CO_OCCUR: học hành vi mua hàng | Kịch bản 1: P→P→P (`kg_metapath.py` dòng 237–250) |
| 6 | **Semantic Walk** | /sɪˈmæn.tɪk wɔːk/ | Walk qua Aisle: học quan hệ danh mục | Kịch bản 2: P→A→P (`kg_metapath.py` dòng 252–281) |
| 7 | **Long-tail** | /lɒŋ teɪl/ | Sản phẩm hiếm, ít xuất hiện | 46.3% sản phẩm < 50 lần, semantic walk cứu vớt long-tail |
| 8 | **Skip-gram** | /skɪp ɡræm/ | Word2Vec: dự đoán context từ target | Dùng Gensim Word2Vec trên walk sequences (`kg_metapath.py` dòng 307–317) |
| 9 | **CSR (Compressed Sparse Row)** | /siː es ɑːr/ | Định dạng ma trận thưa cho adjacency | `_build_adjacency_csr()` (`kg_metapath.py` dòng 155–157) |
| 10 | **Fallback** | /ˈfɔːl.bæk/ | Cơ chế dự phòng khi walk bị đứt | Behavioral fallback về semantic, semantic fallback về behavioral (`kg_metapath.py` dòng 242–281) |
| 11 | **White-box** | /waɪt bɒks/ | Minh bạch, có thể kiểm soát từng bước | Metapath Walk là white-box, khác với black-box của Item2Vec |
| 12 | **High-order Relationship** | /haɪ ˈɔː.dər rɪˈleɪ.ʃən.ʃɪp/ | Quan hệ bậc cao qua nhiều bước | KGMetapath capture được, Item-CF không |

---

## PHẦN 1 — Cấu Trúc Đồ Thị Tri Thức (Knowledge Graph)

### Q1: Knowledge Graph của em có những gì? Phân biệt node và edge.

**Trả lời:**

KG là **Heterogeneous Graph** (đồ thị dị thể) — có nhiều loại node và edge:

**Node (3 loại):**
| Loại Node | Số lượng | Nguồn |
|-----------|----------|-------|
| Product (P) | ~36,181 | `products.parquet` |
| Aisle (A) | ~134 | `aisles.csv` |
| Department (D) | ~21 (sau lọc non-food) | `departments.csv` |

**Edge (3 loại):**
| Loại Edge | Ký hiệu | Nguồn |
|-----------|---------|-------|
| Đồng xuất hiện | P ↔ P (CO_OCCUR) | order_products — pair từng mua cùng |
| Thuộc về lối đi | P → A (BELONGS_TO) | `aisle_id` trong products |
| Aisle thuộc department | A → D (PART_OF) | Cấu trúc phân cấp |

**Dẫn chứng code** (`kg_metapath.py`, dòng 30–42):
```python
class KGMetapathModel:
    """
    Đồ thị KG:
      - Node: Product (~49K), Aisle (134), Department (21)
      - Edge: CO_OCCUR (P↔P, từ co-occurrence), BELONGS_TO (P→A), PART_OF (A→D)
    
    Metapath Walk (chỉ sinh sequence product IDs):
      1. Behavioral (P→P→P): đi theo cạnh CO_OCCUR
      2. Semantic (P→A→P): nhảy qua Aisle để tiếp cận sản phẩm mới
    """
```

---

### Q2: Tại sao gọi là "Heterogeneous" Graph? So với đồ thị đồng nhất (homogeneous) khác gì?

**Trả lời:**

- **Homogeneous graph**: 1 loại node duy nhất (vd: chỉ Product–Product)
- **Heterogeneous graph**: nhiều loại node và edge — như KG của em có P, A, D và 3 loại quan hệ

Tại sao cần Heterogeneous?

Chỉ dùng P–P (co-occurrence) sẽ bỏ qua thông tin ngữ nghĩa: 2 sản phẩm cùng aisle "fresh vegetables" có quan hệ ngữ nghĩa ngay cả khi chưa từng mua cùng nhau.

Bằng cách đưa Aisle vào đồ thị, random walk có thể **nhảy qua Aisle** để khám phá sản phẩm mới trong cùng danh mục → giải quyết một phần cold-start và sparsity.

---

### Q3: Edge CO_OCCUR được xây như thế nào? Có khác với Item-CF không?

**Trả lời:**

Hoàn toàn **dùng lại** logic đếm co-occurrence từ `_numba_ops.py::count_pairs_numba()` — cùng một hàm Numba JIT.

Điểm khác: KGMetapath có thêm **`edge_threshold = 10`** (giống `min_support`) trước khi xây adjacency:

**Dẫn chứng code** (`kg_metapath.py`, dòng 147–151):
```python
mask = counts >= self.params['edge_threshold']
pair_rows = rows[mask]
pair_cols = cols[mask]
pair_counts = counts[mask]
```

Sau đó xây **adjacency CSR** (không phải sparse matrix như Item-CF) bằng `_build_adjacency_csr()` — format này tối ưu cho random walk: truy cập neighbors của node theo index O(1).

**Dẫn chứng code** (`kg_metapath.py`, dòng 154–158):
```python
indptr, neighbors, weights = _build_adjacency_csr(
    pair_rows, pair_cols, pair_counts, n_products
)
self.graph_csr = (indptr, neighbors, weights)
```

---

### Q4: 3 loại edge trong KG là gì? Ý nghĩa?

**Trả lời:**

| Edge | Từ → Đến | Ý nghĩa | Nguồn dữ liệu |
|------|---------|---------|--------------|
| **CO_OCCUR** | P1 → P2 | Hai sản phẩm thường được mua cùng nhau | Ma trận co-occurrence từ 31.9M records |
| **BELONGS_TO** | P → A | Sản phẩm thuộc về aisle nào | File products.csv (aisle_id) |
| **PART_OF** | A → D | Aisle thuộc department nào | File departments.csv |

**Dẫn chứng code** (`kg_metapath.py`, dòng 92–102) — BELONGS_TO:
```python
# Xây mapping KG: Product → Aisle
for _, row in products_df.iterrows():
    pid = row['product_id']
    aid = row['aisle_id']
    if pid in self.product_id_to_idx:
        pidx = self.product_id_to_idx[pid]
        self.product_to_aisle[pidx] = int(aid)
        if int(aid) not in self.aisle_to_products:
            self.aisle_to_products[int(aid)] = []
        self.aisle_to_products[int(aid)].append(pidx)
```

---

## PHẦN 2 — Metapath Walk

### Q5: Metapath là gì? Em có bao nhiêu kịch bản Metapath?

**Trả lời:**

**Metapath** = pattern đường đi định sẵn qua các loại node khác nhau trong Heterogeneous Graph.

Em có **2 kịch bản Metapath**:

**Kịch bản 1 — Behavioral (50%):**
```
P --CO_OCCUR--> P --CO_OCCUR--> P --CO_OCCUR--> P ...
```
→ Đi hoàn toàn qua cạnh CO_OCCUR, chỉ gặp Product nodes. Học hành vi mua kèm thực tế.

**Kịch bản 2 — Semantic (50%):**
```
P --BELONGS_TO--> A --random--> P --BELONGS_TO--> A ...
```
→ Từ Product nhảy lên Aisle, từ Aisle chọn ngẫu nhiên một Product khác cùng lối đi. Học quan hệ danh mục.

**Dẫn chứng code** (`config.py`, dòng 83 và `kg_metapath.py`, dòng 223):
```python
# config.py
MW_METAPATH_BEHAVIORAL_RATIO = 0.5   # 50% Behavioral, 50% Semantic

# Trong fit():
use_behavioral = np.random.random() < behavioral_ratio
```

---

### Q6: Khi chọn kịch bản Behavioral, sản phẩm sẽ nhảy sang sản phẩm nào? Theo tiêu chí gì?

**Trả lời:**

Chọn **ngẫu nhiên đều (uniform random)** trong số các neighbors có edge CO_OCCUR — **không theo trọng số**.

Tuy nhiên, cần nhớ: các neighbors đã được **lọc bởi edge_threshold = 10** — tức là chỉ những sản phẩm đã mua cùng ít nhất 10 lần mới có edge. Nghĩa là neighbors đều là "đủ tin cậy".

Cơ chế chọn neighbor:
```python
s = int(indptr[cur])        # start index trong neighbors array
e = int(indptr[cur + 1])    # end index
n_nb = e - s                # số neighbors
nb_idx = int(rand_matrix[walk_idx, step]) % n_nb   # chọn ngẫu nhiên đều
cur = int(neighbors[s + nb_idx])
```

**Dẫn chứng code** (`kg_metapath.py`, dòng 237–250):
```python
# Behavioral: đi theo CO_OCCUR edge
s = int(indptr[cur])
e = int(indptr[cur + 1])
n_nb = e - s
if n_nb == 0:
    # Hết đường → fallback
    ...
else:
    nb_idx = int(rand_matrix[walk_idx, step]) % n_nb
    cur = int(neighbors[s + nb_idx])
```

---

### Q7: Khi chọn kịch bản Semantic, sản phẩm nhảy sang sản phẩm khác qua Aisle như thế nào?

**Trả lời:**

2 bước:

**Bước 1 — P → A (nhảy lên Aisle):**
```python
aisle_id = self.product_to_aisle.get(cur)
```
→ Mapping đã xây sẵn: mỗi product_idx → aisle_id. Không ngẫu nhiên — mỗi sản phẩm thuộc đúng 1 aisle.

**Bước 2 — A → P (nhảy xuống Product khác cùng Aisle):**
```python
products_in_aisle = self.aisle_to_products.get(aisle_id, [])
candidates = [p for p in products_in_aisle if p != cur]  # loại chính nó
cur = candidates[int(rand_matrix[walk_idx, step]) % len(candidates)]
```
→ Chọn **ngẫu nhiên đều** trong danh sách tất cả sản phẩm cùng aisle (trừ chính nó).

**Fallback**: Nếu aisle không có sản phẩm khác → fallback về Behavioral (đi theo CO_OCCUR).

**Dẫn chứng code** (`kg_metapath.py`, dòng 252–281):
```python
# Semantic: P → A → P
aisle_id = self.product_to_aisle.get(cur)
if aisle_id is None:
    # Fallback về behavioral
    ...
else:
    products_in_aisle = self.aisle_to_products.get(aisle_id, [])
    candidates = [p for p in products_in_aisle if p != cur]
    if candidates:
        cur = candidates[int(rand_matrix[walk_idx, step]) % len(candidates)]
    else:
        # Fallback về behavioral
        ...
```

---

### Q8: Walk length = 25, num_walks = 20. Giải thích ý nghĩa và cách tính tổng số walks.

**Trả lời:**

- `walk_length = 25`: mỗi walk dài 25 bước (25 nodes)
- `num_walks = 20`: mỗi starting node được dùng 20 lần

**Tổng số walks:**
```python
total_walks = max(n_behavioral, n_semantic) * num_walks
```

Ví dụ: nếu có 30,000 product nodes có CO_OCCUR edge, và 35,000 có aisle:
```
total_walks = max(30000, 35000) × 20 = 700,000 walks
```

Tổng số "từ" đưa vào Word2Vec ≈ 700,000 × 25 = **17.5 triệu tokens** → đủ lớn để học embedding chất lượng.

**Dẫn chứng code** (`kg_metapath.py`, dòng 200–203):
```python
n_behavioral = len(nodes_with_edges)
n_semantic = len(nodes_with_aisle)
total_walks = max(n_behavioral, n_semantic) * self.params['num_walks']
walk_length = self.params['walk_length']
```

---

### Q9: Random numbers được sinh như thế nào cho walk? Tại sao pre-generate?

**Trả lời:**

Pre-generate toàn bộ random matrix trước khi walk:

**Dẫn chứng code** (`kg_metapath.py`, dòng 210–215):
```python
max_degree = int(np.max(np.diff(indptr))) if n_behavioral > 0 else 1
rand_matrix = np.random.randint(
    0, max(max_degree, 10000),
    size=(total_walks, walk_length),
    dtype=np.int32
)
```

Lý do pre-generate:
1. Gọi `np.random.randint` 1 lần cho `(700K × 25)` nhanh hơn gọi 17.5M lần bên trong vòng lặp
2. Tái sử dụng cùng random matrix → deterministic nếu fix seed

Dùng trong walk:
```python
nb_idx = int(rand_matrix[walk_idx, step]) % n_nb
```
→ `% n_nb` để đảm bảo index nằm trong range neighbors của node hiện tại.

---

### Q10: Fallback mechanism là gì? Tại sao cần?

**Trả lời:**

**Fallback** là cơ chế dự phòng khi walk bị đứt giữa chừng:

1. **Behavioral walk gặp node không có CO_OCCUR edge:** → Fallback về semantic (chọn node có aisle)
2. **Semantic walk gặp node không có aisle:** → Fallback về behavioral (đi theo CO_OCCUR)
3. **Cả 2 đều không có:** → Dừng walk

**Tại sao cần:** Đảm bảo walk không bị đứt giữa chừng, tạo ra sequences đủ dài để train Word2Vec.

**Dẫn chứng code** (`kg_metapath.py`, dòng 242–281):
```python
# Behavioral fallback
if n_nb == 0:
    if n_semantic > 0:
        cur = int(nodes_with_aisle[walk_idx % n_semantic])
    else:
        break

# Semantic fallback
aisle_id = self.product_to_aisle.get(cur)
if aisle_id is None:
    s = int(indptr[cur])
    e = int(indptr[cur + 1])
    n_nb = e - s
    if n_nb > 0:
        nb_idx = int(rand_matrix[walk_idx, step]) % n_nb
        cur = int(neighbors[s + nb_idx])
    else:
        break
```

---

### Q11: Walk length trung bình là bao nhiêu? Tại sao không đạt 25?

**Trả lời:**

Walk có thể bị đứt sớm vì:
1. Behavioral walk gặp node không có CO_OCCUR edge và không có fallback
2. Semantic walk gặp node không có aisle và behavioral cũng không có edge

Code track `walk_lengths[walk_idx]` để biết độ dài thực tế. Các walk ngắn hơn 25 vẫn được dùng, chỉ bỏ qua các bước `-1`.

**Dẫn chứng code** (`kg_metapath.py`, dòng 286–292):
```python
walk_lengths[walk_idx] = wlen
print(f"  Walk length trung bình: {walk_lengths.mean():.1f}")
```

---

## PHẦN 3 — Từ Walk → Embedding

### Q12: Sau khi có walks, bước tiếp theo là gì?

**Trả lời:**

Walks được chuyển thành **sentences** cho Word2Vec:

**Dẫn chứng code** (`kg_metapath.py`, dòng 295–302):
```python
sentences = []
for i in range(walks.shape[0]):
    wlen = walk_lengths[i]
    if wlen > 1:
        sentence = [str(walks[i, j]) for j in range(wlen)]
        sentences.append(sentence)
```

→ Mỗi walk = 1 "câu", mỗi node (product_idx) được convert sang string.

Sau đó train **Word2Vec Skip-gram + Negative Sampling** trên tập sentences này — hoàn toàn giống Item2Vec nhưng input là walks thay vì đơn hàng gốc.

Parameters: `dim=128, window=10, negative=10, epochs=20`.

**Lưu ý quan trọng**: `window=10` trong KGMetapath lớn hơn `window=5` trong Item2Vec vì walk dài hơn (25 bước) → cần context rộng hơn để bắt được quan hệ xa trong đồ thị.

**Dẫn chứng code** (`kg_metapath.py`, dòng 307–317):
```python
self.model = Word2Vec(
    sentences=sentences,
    vector_size=self.params['embedding_dim'],
    window=self.params['window'],
    min_count=1,    # giữ tất cả product nodes
    negative=self.params['negative'],
    epochs=self.params['epochs'],
    workers=self.params['workers'],
    sg=1,
    seed=RANDOM_SEED,
)
```

---

### Q13: Tại sao min_count = 1 trong Word2Vec của KGMetapath?

**Trả lời:**

**min_count=1** có nghĩa: giữ lại tất cả product nodes, kể cả sản phẩm chỉ xuất hiện 1 lần trong walk sequences.

**Lý do:**
- Mục tiêu của KGMetapath là xử lý long-tail
- Nếu đặt min_count > 1, sản phẩm hiếm sẽ bị loại → mất ý nghĩa của semantic walk
- Walk sequences đã được thiết kế để bao phủ cả sản phẩm hiếm

**Dẫn chứng code** (`kg_metapath.py`, dòng 311):
```python
min_count=1,    # giữ tất cả product nodes
```

---

### Q14: Embedding cuối cùng lưu như thế nào? Recommend dùng gì?

**Trả lời:**

Sau train, extract embeddings từ `model.wv` vào numpy array:

**Dẫn chứng code** (`kg_metapath.py`, dòng 320–329):
```python
self.embeddings = np.zeros((n_products, self.params['embedding_dim']))
for node in range(n_products):
    pid = self.idx_to_product_id[node]
    pid_str = str(pid)
    if pid_str in self.model.wv:
        self.embeddings[node] = self.model.wv[pid_str]

# Cache embedding norms
self._embedding_norms = np.linalg.norm(self.embeddings, axis=1)
self._embedding_norms[self._embedding_norms == 0] = 1e-9
```

Khi recommend: **Cosine similarity vectorized** toàn bộ ma trận:
```python
similarities = (self.embeddings @ vec_a) / (self._embedding_norms * norm_a)
```
→ Matrix multiplication `(n × 128) @ (128,)` = `(n,)` — 1 phép tính cho tất cả sản phẩm, cực nhanh.

**Dẫn chứng code** (`kg_metapath.py`, dòng 358–361):
```python
similarities = (self.embeddings @ vec_a) / (
    self._embedding_norms * norm_a
)
```

---

## PHẦN 4 — Tối Ưu & So Sánh

### Q15: Tại sao dùng Numba cho count_pairs và build_adjacency?

**Trả lời:**

Numba JIT biên dịch Python sang machine code tại runtime, giúp:
- `count_pairs_numba`: Đếm co-occurrence pairs từ 31.9M records nhanh hơn ~100x so với Python thuần
- `_build_adjacency_csr`: Xây adjacency CSR từ edge list nhanh hơn

Cả 2 hàm đều dùng `@njit` decorator và chỉ dùng numpy arrays + primitive types (không dùng Python objects).

**Dẫn chứng code** (`_numba_ops.py`, dòng 16–17 và dòng 121–122):
```python
@njit
def count_pairs_numba(order_indices, order_ptr, n_products):

@njit
def _build_adjacency_csr(pair_rows, pair_cols, pair_counts, n_products):
```

---

### Q16: Tại sao cần cả graph_csr (Numba) và graph (Python dict)?

**Trả lời:**

- **graph_csr (numpy arrays):** Dùng cho Numba JIT functions (nhanh, không dùng Python objects)
- **graph (Python dict):** Dùng cho Python code (dễ đọc, dễ debug)

**Dẫn chứng code** (`kg_metapath.py`, dòng 158–170):
```python
# graph_csr cho Numba
self.graph_csr = (indptr, neighbors, weights)

# graph dict cho Python
graph = {}
for node in range(n_products):
    start = indptr[node]
    end = indptr[node + 1]
    if start < end:
        graph[node] = [...]
```

---

### Q17: KGMetapath khác Item2Vec như thế nào? Học gì thêm?

**Trả lời:**

| Khía cạnh | Item2Vec | KGMetapath |
|-----------|----------|------------|
| Input cho Word2Vec | Đơn hàng gốc | Metapath walks từ KG |
| Quan hệ học được | Co-occurrence trực tiếp | Co-occurrence + quan hệ aisle (semantic) |
| Xử lý sparse items | Kém (ít xuất hiện = ít học) | Tốt hơn (semantic walk khám phá qua aisle) |
| Context window | 5 | 10 |
| Phức tạp | Thấp | Cao hơn (phải xây KG + walk) |

KGMetapath bổ sung thông tin **phân cấp danh mục** (Product → Aisle → Department) vào quá trình học embedding — điều Item2Vec không làm được.

---

### Q18: KGMetapath xử lý long-tail thế nào?

**Trả lời:**

KGMetapath xử lý long-tail bằng **semantic walk**:
- Khi walk gặp sản phẩm hiếm (không có CO_OCCUR edge), thay vì bị đứt, nó đi qua Aisle
- P1 --BELONGS_TO--> A --random--> P2: từ Aisle chọn sản phẩm khác trong cùng aisle
- Giúp sản phẩm hiếm vẫn có embedding và được gợi ý

---

### Q19: KGMetapath so với Item-CF?

**Trả lời:**

| Tiêu chí | Item-CF | KGMetapath |
|----------|---------|------------|
| **Loại** | Memory-based CF | Graph-based (KG embedding) |
| **Dữ liệu** | Chỉ co-occurrence | Co-occurrence + Aisle + Department |
| **Quan hệ** | Bậc 1 (trực tiếp) | Bậc cao (metapath walk) |
| **Long-tail** | ⚠️ Không tốt | ✅ Tốt (semantic walk) |
| **Cold-start** | ❌ | ⚠️ Qua aisle |
| **Giải thích** | ✅ Dễ | ❌ Khó |

---

### Q20: Vai trò của KGMetapath trong Ensemble?

**Trả lời:**

KGMetapath đóng góp **quan hệ bậc cao** và **xử lý long-tail**:
- Item-CF (0.5): Quan hệ trực tiếp, có hướng
- Item2Vec (0.25): Quan hệ gián tiếp
- **KGMetapath (0.25): Quan hệ bậc cao, long-tail, cold-start**

**Dẫn chứng code** (`config.py`, dòng 88–90):
```python
ENS_ALPHA = 0.5   # Item-CF
ENS_BETA = 0.25   # Item2Vec
ENS_GAMMA = 0.25  # KGMetapath
```

---

### Q21: KGMetapath có cold-start không?

**Trả lời:**

**Có thể, nếu biết aisle của sản phẩm.** KGMetapath có thể gợi ý sản phẩm mới qua semantic walk:
- Nếu sản phẩm mới được gán aisle → có thể đi P→A→P
- Nếu không biết aisle → không có mapping → không gợi ý được

Đây là lợi thế so với Item-CF và Item2Vec (hoàn toàn không cold-start).

**Dẫn chứng code** (`kg_metapath.py`, dòng 348–349):
```python
if product_id not in self.product_id_to_idx:
    return []
```

---

### Q22: KGMetapath có phải là white-box không? Tại sao?

**Trả lời:**

**Phần walk là white-box**, phần Word2Vec là black-box:
- **White-box (Metapath Walk):** Có thể kiểm soát và giải thích từng bước: "Từ P1 đi CO_OCCUR đến P2, rồi semantic qua Aisle đến P3"
- **Black-box (Word2Vec):** Skip-gram vẫn là neural network, không giải thích được tại sao 2 embedding gần nhau

---

### Q23: Hạn chế lớn nhất của KGMetapath là gì?

**Trả lời:**

**Hạn chế lớn nhất: Thời gian train chậm.**

KGMetapath phải qua nhiều bước:
1. Xây CSR order_indices (scan 31.9M records)
2. Đếm co-occurrence pairs (Numba)
3. Xây adjacency CSR (Numba)
4. Metapath Walk (20 walks × 25 bước × 36K nodes)
5. Train Word2Vec

Tổng thời gian lâu hơn Item-CF và Item2Vec rất nhiều.

---

### Q24: Nếu được cải thiện KGMetapath, em sẽ làm gì?

**Trả lời:**

3 hướng cải thiện:

1. **Thêm edge type:** Thêm edge giữa Product và Department (P→D) để có thêm đường semantic walk

2. **Tối ưu walk:** Dùng alias sampling thay vì uniform random để ưu tiên edge có trọng số cao hơn

3. **Adaptive behavioral ratio:** Thay vì cố định 0.5, dùng ratio phụ thuộc vào độ phổ biến của sản phẩm:
   - Sản phẩm phổ biến → behavioral nhiều hơn
   - Sản phẩm hiếm → semantic nhiều hơn

**Dẫn chứng code** (`kg_metapath.py`, dòng 223):
```python
use_behavioral = np.random.random() < behavioral_ratio
```

---

## TÓM TẮT LUỒNG THUẬT TOÁN

```
products.parquet + order_products.parquet
        ↓
  Xây mapping: product_idx ↔ aisle_id (BELONGS_TO)
        ↓ (kg_metapath.py dòng 92–102)
  Đếm co-occurrence pairs (Numba) → lọc edge_threshold=10
        ↓ (kg_metapath.py dòng 140–151)
  Xây adjacency CSR (CO_OCCUR graph)
        ↓ (kg_metapath.py dòng 154–158)
  For each walk (700K walks total):
    50%: Behavioral — P→P→P theo CO_OCCUR (uniform random neighbor)
    50%: Semantic  — P→A→P→A→P (A=aisle, chọn product random cùng aisle)
  Walk length = 25 bước
        ↓ (kg_metapath.py dòng 221–286)
  Converts walks → sentences (list of product_idx strings)
        ↓ (kg_metapath.py dòng 295–302)
  Word2Vec Skip-gram + Negative Sampling (min_count=1, window=10, 20 epochs)
        ↓ (kg_metapath.py dòng 307–317)
  Extract W = embeddings (n_products × 128)
        ↓ (kg_metapath.py dòng 320–329)
  Khi recommend:
  Cosine similarity: (embeddings @ v_A) / (norms × norm_A)
  → top-K sản phẩm
        ↓ (kg_metapath.py dòng 358–361)