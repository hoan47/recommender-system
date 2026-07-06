# Dữ Liệu Sau Khi Làm Sạch — Instacart Market Basket Analysis

## 1. Quy Trình Làm Sạch

Dữ liệu gốc Instacart chứa cả sản phẩm thực phẩm (food) và phi thực phẩm (non-food). Để phục vụ bài toán **gợi ý mua kèm thực phẩm**, chúng tôi thực hiện lọc bỏ các sản phẩm non-food dựa trên danh sách các phòng ban (department) không liên quan đến thực phẩm.

### 1.1 Tiêu chí loại bỏ
Các sản phẩm thuộc các phòng ban sau bị loại khỏi dữ liệu:
- **household** — Đồ gia dụng
- **personal care** — Chăm sóc cá nhân
- **pets** — Thú cưng
- **baby** — Đồ cho bé (tã, bỉm...)
- **other** — Khác

### 1.2 Kết quả lọc

| Chỉ số | Giá trị |
|--------|---------|
| Sản phẩm non-food bị loại | **13,507 (27.2%)** |
| Records non-food bị loại | **1,899,791 (6.0%)** |
| **Sản phẩm food giữ lại** | **36,181** |
| **Records food giữ lại** | **31,919,315** |
| **Số đơn hàng sau lọc** | **3,318,066** |

> Dữ liệu sau lọc được lưu dưới dạng Parquet:
> - `products.parquet` (36,181 records) — sản phẩm food
> - `order_products.parquet` (31,919,315 records) — giao dịch food

---

## 2. Cấu Trúc Dữ Liệu Sau Lọc

### 2.1 `products.parquet`
Là nguồn dữ liệu sản phẩm duy nhất cho tất cả models (Item-CF, Item2Vec, KGMetapath, CB Filter, Ensemble).

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `product_id` | int | Mã sản phẩm |
| `product_name` | string | Tên sản phẩm (tiếng Việt — từ `products_vi.csv`) |
| `aisle` | string | Tên lối đi (tiếng Việt — từ `aisles_vi.csv`) |
| `department` | string | Tên phòng ban (tiếng Việt — từ `departments_vi.csv`) |

> **Lưu ý:** Cả 3 cột mô tả đều được ghi đè bằng tiếng Việt. File `aisles.csv` gốc vẫn được dùng riêng cho `product_filter.py` để map `EXCLUDED_DEPARTMENT_NAMES` → `department_id` (không phụ thuộc tên Anh/Việt).

### 2.2 `order_products.parquet`
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | Mã đơn hàng |
| `product_id` | int | Mã sản phẩm |
| `add_to_cart_order` | int | Thứ tự cho vào giỏ hàng |
| `reordered` | int | 1 = đã mua lại, 0 = lần đầu mua |
| `user_id` | int | Mã người dùng (merge từ orders.csv) |
| `order_number` | int | Số thứ tự đơn hàng |
| `order_dow` | int | Ngày trong tuần |
| `order_hour_of_day` | int | Giờ trong ngày |
| `days_since_prior_order` | float | Số ngày kể từ đơn trước |

---

## 3. Thống Kê Dữ Liệu Sau Lọc

### 3.1 Phân bố Eval Set (sau lọc)

| Eval Set | Số lượng | Mục đích |
|----------|----------|----------|
| **prior** | **~3.0 triệu** | Lịch sử mua hàng quá khứ → DÙNG ĐỂ TRAIN |
| **train** | **~131,000** | Dữ liệu train có sẵn → DÙNG ĐỂ TRAIN |
| **test** | **~75,000** | Dữ liệu test có sẵn → **KHÔNG DÙNG để đánh giá gợi ý mua kèm** |

> **Tổng số dòng dữ liệu khả dụng để train:** **31,919,315** (prior + train, đã loại non-food)

### 3.2 Top 10 Lối đi (Aisle) có nhiều sản phẩm nhất

1. **fresh vegetables**
2. **fresh fruits**
3. **packaged vegetables fruits**
4. **yogurt**
5. **packaged cheese**
6. **milk**
7. **water seltzer sparkling water**
8. **chips pretzels**
9. **baby food formula**
10. **bread**

### 3.3 Phân bố Phòng ban (Department)

| Department | Mô tả |
|------------|-------|
| **produce** | Rau củ quả tươi (lớn nhất) |
| **dairy eggs** | Sữa và trứng (lớn thứ 2) |
| **snacks** | Đồ ăn nhẹ |
| **beverages** | Đồ uống |
| **frozen** | Đồ đông lạnh |
| **pantry** | Thực phẩm khô |
| **deli** | Đồ ăn chế biến sẵn |
| ... | ... |

### 3.4 Thống kê Reorder (Mua lại)

Dựa trên mẫu 100,000 dòng từ `order_products__prior.csv`:
- **Tỷ lệ reorder:** ~59-60% (sản phẩm được mua lại)
- **Tỷ lệ mua mới:** ~40-41% (sản phẩm lần đầu mua)

Điều này cho thấy người dùng có xu hướng mua lại các sản phẩm quen thuộc, là tín hiệu tốt cho bài toán gợi ý mua kèm.

### 3.5 Thống kê Kích thước Giỏ hàng (Basket Size)

Dựa trên toàn bộ `order_products__prior.csv` (3,214,874 đơn hàng):

| Chỉ số | Giá trị |
|--------|---------|
| Số đơn hàng | **3,214,874** |
| Trung bình (mean) | **~10.09 sản phẩm/đơn** |
| Độ lệch chuẩn (std) | **7.53** |
| Nhỏ nhất (min) | **1** |
| 25th percentile | **5** |
| Trung vị (50%) | **8** |
| 75th percentile | **14** |
| Lớn nhất (max) | **145** |

> Phân bố lệch phải (right-skewed): đa số đơn hàng có 5–14 sản phẩm, nhưng tồn tại đơn hàng rất lớn (tới 145 sản phẩm). Trung vị 8 cho thấy một nửa số đơn hàng có ≤ 8 sản phẩm.

### 3.6 Thống kê Tần suất Sản phẩm (Product Frequency)

Dựa trên 49,677 sản phẩm trong `order_products__prior.csv`:

| Ngưỡng xuất hiện | Số sản phẩm | Tỷ lệ |
|------------------|-------------|-------|
| < 10 lần | **7,165** | ~14.4% |
| < 30 lần | **17,850** | ~35.9% |
| < 50 lần | **22,991** | ~46.3% |

> **Nhận xét:** Gần một nửa số sản phẩm (46.3%) xuất hiện dưới 50 lần trong toàn bộ tập prior — đây là hiện tượng **long-tail** điển hình. Các sản phẩm hiếm gặp sẽ gây khó khăn cho các phương pháp gợi ý dựa trên đồng xuất hiện (co-occurrence), cần có chiến lược xử lý riêng (ví dụ: fallback sang department/aisle level, hoặc dùng content-based filtering).

---

## 4. Đánh Giá

### 4.1 Ưu điểm
✅ **Kích thước lớn:** 31.9+ triệu giao dịch, đủ để huấn luyện model deep learning  
✅ **Dữ liệu thực tế:** Từ người dùng Instacart thật  
✅ **Đã loại nhiễu:** Non-food được loại bỏ, chỉ giữ lại sản phẩm thực phẩm  
✅ **Cấu trúc rõ ràng:** Quan hệ giữa các bảng được thiết kế tốt  
✅ **Thông tin phong phú:** Bao gồm thời gian, thứ tự, và tần suất mua lại  

### 4.2 Hạn chế
⚠️ **Không có thông tin giá:** Không thể tính toán ngân sách hoặc giá trị đơn hàng  
⚠️ **Không có demographics:** Không biết tuổi, giới tính, vị trí của người dùng  
⚠️ **Dữ liệu test không phù hợp:** `eval_set=test` là dữ liệu tuần tự không phải mua kèm thực tế  

---

## 5. Kết Luận

Sau quá trình làm sạch, tập dữ liệu Instacart còn lại **31,919,315 giao dịch** từ **3,318,066 đơn hàng** và **36,181 sản phẩm thực phẩm**. Dữ liệu này cung cấp đủ độ phong phú để xây dựng model gợi ý mua kèm chất lượng cao.