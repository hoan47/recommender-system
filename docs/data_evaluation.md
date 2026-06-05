# 📊 Đánh giá tập dữ liệu — Instacart Market Basket Analysis

## 1. Tổng quan

Tập dữ liệu sử dụng là **Instacart Market Basket Analysis** — tập dữ liệu công khai chứa lịch sử đơn hàng của người dùng Instacart. Dữ liệu phản ánh hành vi mua sắm thực tế, phù hợp để xây dựng hệ thống gợi ý sản phẩm (recommender system).

## 2. Cấu trúc dữ liệu

### 2.1. File & kích thước

| File | Số dòng (không tính header) | Mô tả |
|------|------------------------------|-------|
| `aisles.csv` | 134 | Danh sách kệ hàng (aisle) |
| `departments.csv` | 21 | Danh sách ngành hàng (department) |
| `products.csv` | 49,688 | Danh sách sản phẩm |
| `orders.csv` | 3,421,083 | Thông tin đơn hàng |
| `order_products__prior.csv` | 32,434,489 | Lịch sử sản phẩm trong đơn (quá khứ) |
| `order_products__train.csv` | 1,384,617 | Label cho huấn luyện |

### 2.2. Schema từng file

**aisles.csv**
| Column | Type | Mô tả |
|--------|------|-------|
| `aisle_id` | int | ID kệ hàng |
| `aisle` | string | Tên kệ hàng |

**departments.csv**
| Column | Type | Mô tả |
|--------|------|-------|
| `department_id` | int | ID ngành hàng |
| `department` | string | Tên ngành hàng |

**products.csv**
| Column | Type | Mô tả |
|--------|------|-------|
| `product_id` | int | ID sản phẩm (PK) |
| `product_name` | string | Tên sản phẩm |
| `aisle_id` | int | FK → aisles |
| `department_id` | int | FK → departments |

**orders.csv**
| Column | Type | Mô tả |
|--------|------|-------|
| `order_id` | int | ID đơn hàng (PK) |
| `user_id` | int | ID người dùng |
| `eval_set` | string | prior / train / test |
| `order_number` | int | Thứ tự đơn của user (1 = đầu tiên) |
| `order_dow` | int | Ngày trong tuần (0 = Chủ nhật) |
| `order_hour_of_day` | int | Giờ trong ngày |
| `days_since_prior_order` | float | Số ngày từ đơn trước |

**order_products__prior.csv / train.csv**
| Column | Type | Mô tả |
|--------|------|-------|
| `order_id` | int | FK → orders |
| `product_id` | int | FK → products |
| `add_to_cart_order` | int | Thứ tự cho vào giỏ |
| `reordered` | int | 1 = đã mua lại, 0 = mua lần đầu |

## 3. Thống kê chính

### 3.1. Người dùng & Đơn hàng

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số người dùng | **206,209** |
| Tổng số đơn hàng | **3,421,083** |
| Đơn hàng trung bình / user | **16.6** (median: 10, min: 4, max: 100) |

**Phân bố `eval_set`:**

| eval_set | Số lượng | % |
|----------|----------|---|
| `prior` (lịch sử) | 3,214,874 | 94.0% |
| `train` (huấn luyện) | 131,209 | 3.8% |
| `test` (đánh giá) | 75,000 | 2.2% |

### 3.2. Sản phẩm & Danh mục

| Chỉ số | Giá trị |
|--------|---------|
| Tổng sản phẩm | **49,688** |
| Số ngành hàng (department) | **21** |
| Số kệ hàng (aisle) | **134** |
| Sản phẩm xuất hiện trong prior | **49,677** / 49,688 |

### 3.3. Hành vi mua hàng

| Chỉ số | Giá trị |
|--------|---------|
| Số sản phẩm / đơn (avg) | **10.1** |
| Số sản phẩm / đơn (median) | **8** |
| Số sản phẩm / đơn (min–max) | **1 – 145** |
| Tỉ lệ reorder (prior) | **59.0%** |
| Tỉ lệ reorder (train) | **59.9%** |
| Tổng tương tác (prior) | **32,434,489** |
| Tổng tương tác (train) | **1,384,617** |

## 4. Nhận xét & Hệ quả cho Recommender System

### 4.1. Dữ liệu đủ lớn & Representative

- 206K users + 50K products + 32M interactions là kích thước chuẩn cho bài toán market basket.
- 21 departments giúp phân tích cross-department recommendation.
- Mỗi user có **ít nhất 4 đơn hàng** — đủ lịch sử cho personalized/sequential recommendation.

### 4.2. Sparsity (độ thưa)

- Mật độ ma trận user–item ≈ 0.3% (32M / 206K×50K × 100).
- Cần xử lý sparsity: collaborative filtering cần similarity measures phù hợp, hoặc kết hợp Content-Based.

### 4.3. Imbalance reorder

- ~59% reorder vs ~41% first-time purchase → model có thiên hướng predict "reorder" nhiều hơn.
- Cần dùng metrics phù hợp như NDCG, MAP thay vì accuracy thuần túy.
- Có thể cân nhắc weighted loss hoặc sampling strategies.

### 4.4. Long-tail products

- Nhiều sản phẩm có tần suất xuất hiện thấp (long-tail).
- **Content-Based filtering** hữu ích cho cold-start và long-tail items dựa trên TF-IDF từ tên sản phẩm, aisle, department.
- **Hybrid approach** (kết hợp CB + CF + KG) như mô tả trong `docs/models.md` là hướng đi hợp lý.

### 4.5. Temporal patterns

- `order_hour_of_day`, `order_dow`, `days_since_prior_order` cho phép phân tích hành vi theo thời gian.
- Có thể xây dựng tính năng temporal features cho model.

### 4.6. Chia tập hợp lý

- prior: dữ liệu lịch sử → dùng để xây dựng user profiles, item similarities, knowledge graph.
- train: 131K đơn cuối của mỗi user → ground truth cho huấn luyện.
- test: 75K đơn → đánh giá model.

### 4.7. Lưu ý kỹ thuật

- File products.csv chứa ký tự đặc biệt (em dash, accented characters) → cần đọc với encoding `utf-8`.
- Một số tên sản phẩm có chứa dấu phẩy trong ngoặc kép → cần dùng `csv.DictReader` thay vì split bằng dấu phẩy thủ công.

## 5. Tóm tắt

| Khía cạnh | Đánh giá |
|-----------|----------|
| ✅ Kích thước | Phù hợp, đủ lớn để train deep learning models |
| ✅ Chất lượng | Dữ liệu sạch, có cấu trúc rõ ràng |
| ✅ Đa dạng | 50K sản phẩm, 21 departments, 134 aisles |
| ✅ Temporal | Có timestamp, thứ tự đơn hàng |
| ✅ Split sẵn | prior / train / test rõ ràng |
| ⚠️ Sparsity | Cần xử lý (dùng hybrid / item-based CF) |
| ⚠️ Imbalance | 59% reorder — cần metric phù hợp |

Dataset này hoàn toàn phù hợp để xây dựng hệ thống gợi ý sản phẩm với 4 models đã định nghĩa: **Content-Based**, **Collaborative Filtering (SPMI)**, **Knowledge Graph (RWR)**, và **Hybrid**.