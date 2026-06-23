# Khảo Sát Tập Dữ Liệu Instacart Market Basket Analysis

## 1. Tổng Quan

Tập dữ liệu **Instacart Market Basket Analysis** là bộ dữ liệu công khai từ Instacart, chứa thông tin về các đơn hàng tạp hóa trực tuyến. Dữ liệu này được thu thập từ người dùng thực tế và đã được ẩn danh hóa.

### 1.1 Nguồn gốc
- **Nguồn:** Instacart (nền tảng giao hàng tạp hóa trực tuyến)
- **Mục đích ban đầu:** Phân tích giỏ hàng (Market Basket Analysis)
- **Kích thước:** Cực kỳ lớn, tổng cộng **~37.3 triệu dòng dữ liệu** trên 6 file CSV

---

## 2. Cấu Trúc Dữ Liệu Chi Tiết

### 2.1 File: `orders.csv`
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | Mã đơn hàng (duy nhất) |
| `user_id` | int | Mã người dùng |
| `eval_set` | string | Phân loại: `prior` / `train` / `test` |
| `order_number` | int | Số thứ tự đơn hàng của user (1 = đầu tiên) |
| `order_dow` | int | Ngày trong tuần (0 = Chủ nhật, 6 = Thứ 7) |
| `order_hour_of_day` | int | Giờ trong ngày (0-23) |
| `days_since_prior_order` | float | Số ngày kể từ đơn trước (NaN = đơn đầu tiên) |

### 2.2 File: `products.csv`
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `product_id` | int | Mã sản phẩm |
| `product_name` | string | Tên sản phẩm |
| `aisle_id` | int | Mã lối đi (liên kết với aisles.csv) |
| `department_id` | int | Mã phòng ban (liên kết với departments.csv) |

### 2.3 File: `order_products__prior.csv`
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `order_id` | int | Mã đơn hàng |
| `product_id` | int | Mã sản phẩm |
| `add_to_cart_order` | int | Thứ tự cho vào giỏ hàng |
| `reordered` | int | 1 = đã mua lại sản phẩm này, 0 = lần đầu mua |

### 2.4 File: `order_products__train.csv`
Cấu trúc giống hệt `order_products__prior.csv`, dùng cho tập train.

### 2.5 File: `aisles.csv`
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `aisle_id` | int | Mã lối đi |
| `aisle` | string | Tên lối đi (ví dụ: "fresh fruits", "dairy eggs") |

### 2.6 File: `departments.csv`
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `department_id` | int | Mã phòng ban |
| `department` | string | Tên phòng ban (ví dụ: "produce", "dairy eggs") |

---

## 3. Thống Kê Cơ Bản

### 3.1 Kích thước dữ liệu

| File | Số dòng (bao gồm header) | Kích thước ước tính |
|------|--------------------------|---------------------|
| `orders.csv` | **3,421,084** | ~200 MB |
| `products.csv` | **49,690** | ~3 MB |
| `order_products__prior.csv` | **32,434,491** | ~2.1 GB |
| `order_products__train.csv` | **1,384,619** | ~90 MB |
| `aisles.csv` | **135** | ~5 KB |
| `departments.csv` | **22** | ~1 KB |
| **Tổng cộng** | **~37,290,041** | **~2.4 GB** |

### 3.2 Thống kê Đơn hàng (`orders.csv`)

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số đơn hàng | **3,421,084** |
| Tổng số người dùng | **206,209** |
| Số đơn hàng trung bình/user | ~16.6 |
| Số đơn hàng tối đa/user | 100 |

### 3.3 Phân bố Eval Set

| Eval Set | Số lượng | % | Mục đích |
|----------|----------|---|----------|
| **prior** | **~3.2 triệu** | ~94% | Lịch sử mua hàng quá khứ → DÙNG ĐỂ TRAIN |
| **train** | **~131,000** | ~4% | Dữ liệu train có sẵn → DÙNG ĐỂ TRAIN |
| **test** | **~75,000** | ~2% | Dữ liệu test có sẵn → **KHÔNG DÙNG để đánh giá gợi ý mua kèm** |

> ⚠️ **Lưu ý quan trọng:** Tổng cộng có **31,919,315 dòng** dữ liệu sau khi lọc non-food (từ `order_products__prior.csv` + `order_products__train.csv`) sẽ được sử dụng để train model. Đã loại 1,899,791 records non-food.

### 3.4 Thống kê Sản phẩm

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số sản phẩm gốc | **49,689** |
| Sản phẩm non-food bị loại | **13,507 (27.2%)** |
| **Sản phẩm food giữ lại** | **36,181** |
| Tổng số lối đi (aisles) | **134** |
| Tổng số phòng ban (departments) | **21** |

### 3.5 Top 10 Lối đi (Aisle) có nhiều sản phẩm nhất

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

### 3.6 Phân bố Phòng ban (Department)

| Department | Mô tả |
|------------|-------|
| **produce** | Rau củ quả tươi (lớn nhất) |
| **dairy eggs** | Sữa và trứng (lớn thứ 2) |
| **snacks** | Đồ ăn nhẹ |
| **beverages** | Đồ uống |
| **frozen** | Đồ đông lạnh |
| **pantry** | Thực phẩm khô |
| **deli** | Đồ ăn chế biến sẵn |
| **household** | Đồ gia dụng |
| ... | ... |

### 3.7 Thống kê Reorder (Mua lại)

Dựa trên mẫu 100,000 dòng từ `order_products__prior.csv`:
- **Tỷ lệ reorder:** ~59-60% (sản phẩm được mua lại)
- **Tỷ lệ mua mới:** ~40-41% (sản phẩm lần đầu mua)

Điều này cho thấy người dùng có xu hướng mua lại các sản phẩm quen thuộc, là tín hiệu tốt cho bài toán gợi ý mua kèm.

### 3.8 Thống kê Kích thước Giỏ hàng (Basket Size)

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

### 3.9 Thống kê Tần suất Sản phẩm (Product Frequency)

Dựa trên 49,677 sản phẩm trong `order_products__prior.csv`:

| Ngưỡng xuất hiện | Số sản phẩm | Tỷ lệ |
|------------------|-------------|-------|
| < 10 lần | **7,165** | ~14.4% |
| < 30 lần | **17,850** | ~35.9% |
| < 50 lần | **22,991** | ~46.3% |

> **Nhận xét:** Gần một nửa số sản phẩm (46.3%) xuất hiện dưới 50 lần trong toàn bộ tập prior — đây là hiện tượng **long-tail** điển hình. Các sản phẩm hiếm gặp sẽ gây khó khăn cho các phương pháp gợi ý dựa trên đồng xuất hiện (co-occurrence), cần có chiến lược xử lý riêng (ví dụ: fallback sang department/aisle level, hoặc dùng content-based filtering).

---

## 4. Phân Tích Chi Tiết

### 4.1 Đặc điểm Dữ liệu Mua kèm

Dữ liệu được cấu trúc theo **đơn hàng (order)**, mỗi đơn hàng chứa nhiều sản phẩm. Đây là định dạng lý tưởng cho:

1. **Phân tích giỏ hàng (Market Basket Analysis):** Xác định sản phẩm nào thường xuất hiện cùng nhau
2. **Gợi ý mua kèm (Bundle Recommendation):** Dựa trên các sản phẩm trong cùng giỏ hàng
3. **Phân tích trình tự mua hàng (Sequential Pattern):** Dựa trên `order_number` và `days_since_prior_order`

### 4.2 Thông tin Bổ sung về Thời gian

- **Order DOW (Ngày trong tuần):** Cho biết người dùng thường mua sắm vào ngày nào
- **Order Hour (Giờ):** Cho biết khung giờ mua sắm phổ biến
- **Days Since Prior:** Khoảng cách giữa các lần mua hàng, giúp xác định tần suất

### 4.3 Thông tin Bổ sung về Sản phẩm

- Phân cấp: **Product → Aisle → Department**
- Tên sản phẩm chi tiết giúp xác định mối quan hệ ngữ nghĩa giữa các sản phẩm
- Có cả sản phẩm thương hiệu và sản phẩm generic

---

## 5. Đánh Giá Chất Lượng Dữ Liệu

### 5.1 Ưu điểm
✅ **Kích thước lớn:** 31.9+ triệu giao dịch (sau lọc non-food), đủ để huấn luyện model deep learning  
✅ **Dữ liệu thực tế:** Từ người dùng Instacart thật  
✅ **Cấu trúc rõ ràng:** Quan hệ giữa các bảng được thiết kế tốt  
✅ **Thông tin phong phú:** Bao gồm thời gian, thứ tự, và tần suất mua lại  

### 5.2 Hạn chế
> **Lưu ý về dữ liệu sản phẩm:**
> - `products.parquet` (36,181 records — đã lọc non-food) — tạo từ `01_load_data.py`. Cả 3 cột mô tả đều được ghi đè bằng **tiếng Việt**:
>   - `product_name` ← từ `products_vi.csv`
>   - `aisle` ← từ `aisles_vi.csv`
>   - `department` ← từ `departments_vi.csv`
> - Dùng cho **tất cả models** (Item-CF, Item2Vec, KGMetapath, CB Filter, Ensemble).
> - Không còn 2 nguồn riêng biệt nữa — `products.parquet` là nguồn duy nhất cho cả collaborative và content-based.
> - File `aisles.csv` (gốc) vẫn được dùng riêng cho `product_filter.py` để map `EXCLUDED_DEPARTMENT_NAMES` → `department_id` (không phụ thuộc tên Anh/Việt).

⚠️ **Không có thông tin giá:** Không thể tính toán ngân sách hoặc giá trị đơn hàng  
⚠️ **Không có demographics:** Không biết tuổi, giới tính, vị trí của người dùng  
⚠️ **Dữ liệu test không phù hợp:** `eval_set=test` là dữ liệu tuần tự không phải mua kèm thực tế  

---

## 6. Kết Luận

Tập dữ liệu Instacart là nguồn tài nguyên tuyệt vời cho bài toán **gợi ý mua kèm**. Với hơn **31.9 triệu giao dịch (sau lọc non-food)** từ **206,000+ người dùng** và **36,181 sản phẩm thực phẩm**, dữ liệu này cung cấp đủ độ phong phú để xây dựng model chất lượng cao.

**Tổng số dòng dữ liệu khả dụng để train:** **31,919,315** (prior + train, đã loại non-food)

---