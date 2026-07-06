# Khảo Sát Tập Dữ Liệu Instacart Market Basket Analysis — Dữ Liệu Gốc

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

## 3. Thống Kê Cơ Bản (Dữ Liệu Gốc)

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
| **prior** | **~3.2 triệu** | ~94% | Lịch sử mua hàng quá khứ |
| **train** | **~131,000** | ~4% | Dữ liệu train có sẵn |
| **test** | **~75,000** | ~2% | Dữ liệu test có sẵn |

### 3.4 Thống kê Sản phẩm

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số sản phẩm gốc | **49,689** |
| Tổng số lối đi (aisles) | **134** |
| Tổng số phòng ban (departments) | **21** |

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

## 5. Đánh Giá Chất Lượng Dữ Liệu Gốc

### 5.1 Ưu điểm
✅ **Kích thước lớn:** Hàng chục triệu giao dịch, đủ để huấn luyện model deep learning  
✅ **Dữ liệu thực tế:** Từ người dùng Instacart thật  
✅ **Cấu trúc rõ ràng:** Quan hệ giữa các bảng được thiết kế tốt  
✅ **Thông tin phong phú:** Bao gồm thời gian, thứ tự, và tần suất mua lại  

### 5.2 Hạn chế
⚠️ **Không có thông tin giá:** Không thể tính toán ngân sách hoặc giá trị đơn hàng  
⚠️ **Không có demographics:** Không biết tuổi, giới tính, vị trí của người dùng  
⚠️ **Dữ liệu test không phù hợp:** `eval_set=test` là dữ liệu tuần tự không phải mua kèm thực tế  
⚠️ **Có sản phẩm non-food:** Cần lọc để chỉ giữ lại sản phẩm thực phẩm phục vụ bài toán gợi ý mua kèm