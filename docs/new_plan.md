# PROMPT: Nâng cấp kỹ thuật viết code — Dự án `recommender-system`

> **Mục tiêu:** Áp dụng các kỹ thuật viết code từ Dự án Phụ vào Dự án Chính.
> Logic nghiệp vụ và model **giữ nguyên hoàn toàn**. Chỉ cải thiện cách viết code.

---

## Bối cảnh

Dự án Chính (`recommender-system`) là một hệ thống gợi ý sản phẩm gồm các file:
```
src/config.py
src/features/build_tfidf.py
src/features/build_spmi.py
src/features/build_knowledge_graph.py
src/features/build_hybrid.py
src/evaluation/evaluate.py
src/utils/data_loader.py
```

Dự án Phụ là một bản thử nghiệm nhỏ hơn (3 file: `all_model.py`, `data_loader.py`, `test.py`) có phong cách viết code tốt hơn ở một số mặt kỹ thuật. Dưới đây là danh sách các kỹ thuật đó cần được áp dụng vào Dự án Chính.

---

## Các kỹ thuật cần áp dụng

---

### 1. Ép kiểu dtype ngay khi load CSV (Downcasting)

**Dự án Phụ làm:**
Có một hàm `_cast_tx()` chuyên biệt, được gọi ngay sau `pd.read_csv()`. Hàm này ép toàn bộ các cột số sang dtype nhỏ nhất phù hợp, có kiểm tra `if col in df.columns` để an toàn với các file không có đủ cột:

```python
def _cast_tx(df: pd.DataFrame):
    casts = {
        'order_id':          np.int32,
        'product_id':        np.int32,
        'user_id':           np.int32,
        'add_to_cart_order': np.int8,
        'reordered':         np.int8,
    }
    for col, dtype in casts.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
```

**Dự án Chính hiện tại:**
Không có hàm tập trung. Một số chỗ ép kiểu inline rải rác, một số chỗ không ép kiểu, để pandas giữ nguyên `int64` mặc định.

**Cần làm:**
- Thêm hàm `_cast_tx()` vào `src/utils/data_loader.py`.
- Gọi ngay sau mỗi lần `pd.read_csv()` trong toàn bộ `data_loader.py`.

---

### 2. RAM Monitor tích hợp trong log

**Dự án Phụ làm:**
Mọi dòng log đều kèm mức RAM hiện tại, giúp theo dõi memory leak ngay trong quá trình chạy:

```python
import os, psutil

def ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

def log(msg):
    print(f"  [RAM {ram_mb():>6.0f} MB]  {msg}")
```

Dùng thay cho `print()` ở tất cả các bước quan trọng:
```python
log(f"prior {len(prior):,} | products {len(products):,}")
log(f"After clean: {len(prior_clean):,} rows")
log("Data ready.")
```

**Dự án Chính hiện tại:**
Dùng `print()` thuần, không có thông tin RAM.

**Cần làm:**
- Thêm `ram_mb()` và `log()` vào `src/utils/data_loader.py`.
- Thay các `print()` ở các bước load/filter lớn bằng `log()`.

---

### 3. Giải phóng bộ nhớ chủ động: `del` + `gc.collect()`

**Dự án Phụ làm:**
Xóa biến trung gian ngay sau khi không cần nữa, rồi gọi `gc.collect()` ngay:

```python
# Xóa ngay sau bước filter
del prior; gc.collect()

# Xóa nhiều biến cùng lúc trên một dòng
del prior_clean, prior_f_all, mask2, mask3; gc.collect()

# Xóa sau khi build xong một cấu trúc dữ liệu lớn
del doc_tokens, doc_freq, vocab, idf
gc.collect()

# Xóa trong vòng lặp lớn để tránh tích lũy
del prior_original; gc.collect()
```

**Dự án Chính hiện tại:**
Ít dùng `del` chủ động, `gc.collect()` chưa được gọi sau các bước tạo cấu trúc lớn.

**Cần làm:**
- Thêm `import gc` vào các file còn thiếu.
- Sau mỗi biến trung gian lớn (DataFrame tạm, dict lớn, ma trận) không còn dùng → thêm `del` + `gc.collect()`.

---

### 4. Prune định kỳ trong vòng lặp lớn

**Dự án Phụ làm:**
Khi xây `co_counts` qua hàng triệu orders, cứ mỗi `_CHUNK` iterations lại prune những entry dưới ngưỡng, tránh dict phình to vô hạn:

```python
_CHUNK     = 50_000
_PRUNE_MIN = 3

for i, prods in enumerate(order_grp):
    # ... xử lý ...
    if (i + 1) % _CHUNK == 0:
        co_counts = defaultdict(np.int16,
                                {k: v for k, v in co_counts.items()
                                 if v >= _PRUNE_MIN})
        gc.collect()
```

**Dự án Chính hiện tại:**
Vòng lặp co-occurrence chạy thẳng đến hết, không có bước prune giữa chừng.

**Cần làm:**
- Thêm hằng số `_CHUNK = 50_000` và `_PRUNE_MIN = 3` vào `build_spmi.py`.
- Thêm block prune mỗi `_CHUNK` iterations trong vòng lặp tính co-occurrence.

---

### 5. Dùng dtype nhỏ cho accumulator dict

**Dự án Phụ làm:**
Khai báo `defaultdict` với dtype numpy nhỏ thay vì để Python tự chọn `int` (64-bit):

```python
co_counts = defaultdict(np.int16)   # Co-occurrence counts: max ~32K, int16 là đủ
oc_counts = defaultdict(np.int32)   # Order counts: có thể lớn hơn, dùng int32
```

**Dự án Chính hiện tại:**
Dùng `defaultdict(int)` — Python `int` không giới hạn kích thước, tốn bộ nhớ gấp nhiều lần.

**Cần làm:**
- Trong `build_spmi.py`, đổi `defaultdict(int)` thành `defaultdict(np.int16)` cho `co_counts` và `defaultdict(np.int32)` cho `oc_counts`.

---

### 6. Căn chỉnh dấu `=` theo cột (Visual alignment)

**Dự án Phụ làm:**
Căn thẳng dấu `=` khi khai báo nhiều biến liên quan, giúp đọc và so sánh nhanh hơn:

```python
# Khai báo biến toàn cục
prior_f        = None
frequent_items = None
test_cases     = None
products       = None
dept_name      = None
prod_dept_map  = None
name_map       = None

# Trong vòng lặp
co_counts = defaultdict(np.int16)
oc_counts = defaultdict(np.int32)

# Trong tính toán
p_ab = cnt / N_est
p_a  = oc_counts[a] / N_est
p_b  = oc_counts[b] / N_est
pmi  = math.log2(p_ab / (p_a * p_b + 1e-12) + 1e-12)
spmi = max(pmi - math.log2(KG_SPMI_SHIFT), 0.0)
```

**Dự án Chính hiện tại:**
Dấu `=` không căn chỉnh, mỗi dòng viết tự do.

**Cần làm:**
- Áp dụng visual alignment cho các block khai báo biến liên quan, tham số tính toán, và assignment trong cùng một nhóm logic.
- Không cần áp dụng toàn bộ file, chỉ áp dụng ở những chỗ có từ 3 biến trở lên trong cùng một block.

---

### 7. Tách hàm nhỏ có prefix `_` cho logic nội bộ

**Dự án Phụ làm:**
Chia logic lớn thành nhiều hàm private nhỏ có prefix `_`, mỗi hàm chỉ làm một việc:

```python
def _init_dept_map():     ...   # khởi tạo mapping product→dept
def _build_cb_vectors():  ...   # xây CB vectors
def _build_stats_and_cooccurrence(): ...
def _build_spmi_edges(co_counts, oc_counts): ...
def _build_graph(spmi_edges_dict): ...

def build_all():          # hàm public duy nhất, gọi các hàm _ theo thứ tự
    _init_dept_map()
    _build_cb_vectors()
    co_counts, oc_counts = _build_stats_and_cooccurrence()
    spmi_dict = _build_spmi_edges(co_counts, oc_counts)
    del co_counts, oc_counts; gc.collect()
    _build_graph(spmi_dict)
```

**Dự án Chính hiện tại:**
Một số file có hàm dài, gộp nhiều bước vào một hàm, khó test từng phần.

**Cần làm:**
- Rà soát các hàm dài (>50 dòng) trong `build_spmi.py`, `build_knowledge_graph.py`, `build_hybrid.py`.
- Tách thành các hàm `_tên_bước()` nhỏ hơn.
- Giữ một hàm public entry point (ví dụ `build_spmi_model()`) gọi các hàm private theo thứ tự.

---

### 8. Separator comment phân vùng rõ ràng

**Dự án Phụ làm:**
Dùng comment dạng banner để phân chia các section lớn trong file, kèm giải thích ngắn:

```python
# ── Bien toan cuc ─────────────────────────────────────────────────────────────
prior_f        = None
...

# =============================================================================
#  BUOC 1 -- Content-Based (Custom TF-IDF)
# =============================================================================
def _build_cb_vectors():
    ...

# =============================================================================
#  BUOC 2 -- Co-occurrence + SPMI
# =============================================================================
def _build_stats_and_cooccurrence():
    ...
```

**Dự án Chính hiện tại:**
Đã dùng `# ===` separator nhưng không nhất quán — một số chỗ dùng, một số chỗ không.

**Cần làm:**
- Chuẩn hóa separator thành một kiểu duy nhất xuyên suốt toàn bộ dự án.
- Mọi section lớn (nhóm hàm, bước xử lý) đều phải có separator.

---

## Tóm tắt — Checklist cho từng file

| File | Việc cần làm |
|---|---|
| `data_loader.py` | Thêm `_cast_tx()`, `ram_mb()`, `log()`; thêm `del` + `gc` sau mỗi bước filter |
| `build_spmi.py` | Đổi `defaultdict(int)` → dtype nhỏ; thêm prune định kỳ; tách hàm `_` nhỏ; visual alignment |
| `build_tfidf.py` | Thêm `del` + `gc` sau bước fit; visual alignment trong assignment block |
| `build_knowledge_graph.py` | Tách hàm `_` nhỏ; thêm separator; thêm `gc` sau các bước build |
| `build_hybrid.py` | Tách hàm `_` nhỏ; visual alignment |
| Tất cả các file | Chuẩn hóa separator comment thành một kiểu duy nhất |

---

## Ràng buộc

- **Không thay đổi** bất kỳ logic nghiệp vụ, công thức tính toán, hay tham số model nào.
- **Không thay đổi** interface của các hàm public (tên hàm, tham số đầu vào, giá trị trả về).
- **Không thay đổi** format file output (`.npz`, `.json`, `.md`).
- Chỉ thay đổi cách tổ chức và viết code bên trong.
