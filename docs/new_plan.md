# Kế hoạch nâng cấp kỹ thuật viết code — Dự án `recommender-system`

> **Mục tiêu:** Áp dụng các kỹ thuật viết code tốt hơn, cải thiện hiệu năng bộ nhớ và khả năng đọc code.
> Logic nghiệp vụ và model **giữ nguyên hoàn toàn**. Chỉ cải thiện cách viết code.

---

## Phân tích hiện trạng (sau khi đọc tất cả file)

| File | Có sẵn `_` hàm | Separator | `del`+`gc` | RAM log |
|------|:---:|:---:|:---:|:---:|
| `data_loader.py` | ❌ | ❌ | ❌ | ❌ |
| `build_spmi.py` | ❌ (hàm dài) | ⚠️ chưa đủ | ❌ | ❌ |
| `build_tfidf.py` | ✅ tốt | ⚠️ một vài chỗ | ❌ | ❌ |
| `build_knowledge_graph.py` | ✅ tốt | ✅ tốt | ❌ | ❌ |
| `build_hybrid.py` | ✅ khá | ❌ | ❌ | ❌ |
| `evaluate.py` | ✅ khá | ✅ | ❌ | ❌ |
| `config.py` | N/A | ✅ | N/A | N/A |

---

## Điều chỉnh so với kế hoạch gốc

### Bị LOẠI BỎ:

| # | Mục gốc | Lý do |
|---|---------|-------|
| 4 | Prune định kỳ `co_counts` | Code dùng `scipy.sparse.lil_matrix`, không phải `defaultdict` — không prune được giữa chừng do ma trận có kích thước cố định `n_products × n_products` |
| 5 | `defaultdict(np.int16)` | Code không dùng `defaultdict` cho co-occurrence — dùng `lil_matrix` với `np.float64`, đã là kiểu phù hợp |

### Được THAY THẾ:

| # | Thay thế | Áp dụng cho |
|---|---------|-------------|
| 4+5 | Thêm `del` + `gc.collect()` sau khi build xong ma trận lớn (cooc, TF-IDF, similarity...) thay vì prune giữa vòng lặp | `build_spmi.py`, `build_tfidf.py` |

---

## Các bước thực hiện (theo thứ tự)

### BƯỚC 0: Cài đặt dependency mới
- Thêm `psutil` vào `requirements.txt`

### BƯỚC 1: `src/utils/data_loader.py` (nền tảng, làm trước)
1. Thêm `import os, gc, psutil, numpy as np`
2. Thêm hàm `_cast_tx(df)` — ép dtype int32/int8 cho các cột
3. Thêm hàm `ram_mb()` + `log()` — log kèm RAM
4. Thay `print()` bằng `log()` ở các bước load/filter
5. Gọi `_cast_tx()` ngay sau mỗi `pd.read_csv()`
6. Thêm `del` + `gc.collect()` sau các bước load/filter lớn
7. Thêm separator cho các nhóm hàm

### BƯỚC 2: `src/features/build_spmi.py`
1. Thêm `import gc`
2. Thêm separator cho các section còn thiếu
3. Tách `count_cooccurrence()` (~60 dòng) thành các hàm `_` nhỏ:
   - `_build_cooc_from_grouped(grouped, n_products)` — logic dựng ma trận
4. Tách `compute_spmi()` (~60 dòng) thành:
   - `_compute_spmi_row(cooc_row, freq_i, order_freqs, total_orders, log_shift)` — tính 1 dòng
5. Visual alignment cho các block khai báo ≥3 biến
6. Thêm `del` + `gc.collect()` sau mỗi bước nặng

### BƯỚC 3: `src/features/build_tfidf.py`
1. Thêm `import gc`
2. Thêm separator cho các nhóm hàm
3. Visual alignment cho block có ≥3 biến
4. Thêm `del` + `gc.collect()` sau các bước fit lớn

### BƯỚC 4: `src/features/build_knowledge_graph.py`
1. Thêm `import gc`
2. Thêm `del` + `gc.collect()` sau các bước: sinh walks, train xong, similarity xong

### BƯỚC 5: `src/features/build_hybrid.py`
1. Thêm `import gc`
2. Thêm separator cho section
3. Visual alignment cho grid search params
4. Thêm `del` + `gc.collect()` sau grid search

### BƯỚC 6: `src/evaluation/evaluate.py`
1. Thêm `import gc`
2. Thêm `del` + `gc.collect()` sau khi load và evaluate từng model

### BƯỚC 7: `src/config.py`
- Không cần thay đổi (đã có separator tốt)

---

## Ràng buộc

- **Không thay đổi** logic nghiệp vụ, công thức tính toán, tham số model
- **Không thay đổi** interface hàm public (tên, tham số, giá trị trả về)
- **Không thay đổi** format file output (`.npz`, `.json`, `.md`)
- Chỉ thay đổi cách tổ chức và viết code bên trong