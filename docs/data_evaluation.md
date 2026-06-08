# 📊 Đánh giá tập dữ liệu — Instacart Market Basket Analysis (Global / Item-Oriented)

## 1. Tổng quan

Tập dữ liệu sử dụng là **Instacart Market Basket Analysis** — tập dữ liệu công khai chứa lịch sử đơn hàng của người dùng Instacart. Dữ liệu phản ánh hành vi mua sắm thực tế, phù hợp để xây dựng hệ thống gợi ý sản phẩm (recommender system).

**Hướng tiếp cận: Global (Item-Oriented)**
- Không xây dựng profile riêng cho từng user
- Thay vào đó, xây dựng **ma trận quan hệ giữa các sản phẩm** dựa trên co-occurrence (Confidence unified scoring), cấu trúc đồ thị (KG), và nội dung sản phẩm (CB)
- Output là item-item similarity/scores — mọi user đều dùng chung một ma trận

## 2. Cấu trúc dữ liệu

### 2.1. File & kích thước

| File | Số dòng (không tính header) | Mô tả |
|------|------------------------------|-------|
| `departments.csv` | 21 | Danh sách ngành hàng (department) |
| `products.csv` | 49,688 | Danh sách sản phẩm |
| `orders.csv` | 3,421,083 | Thông tin đơn hàng |
| `order_products__prior.csv` | 32,434,489 | Tương tác của prior orders — **nguồn chính xây model** |
| `order_products__train.csv` | 1,384,617 | Ground truth cho **CẢ train (131K đơn) VÀ test (75K đơn)** |

> **Ghi chú quan trọng:**
> - File `aisles.csv` (kệ hàng) không được sử dụng. Aisle là vị trí vật lý khác nhau giữa các cửa hàng trong cùng chuỗi, không phải phân loại sản phẩm mang tính toàn cục.
> - **KHÔNG có file `order_products__test.csv` riêng.** Dataset Instacart chỉ cung cấp 2 file interaction: `order_products__prior.csv` và `order_products__train.csv`. Ground truth cho test nằm trong `order_products__train.csv` và được phân biệt qua cột `eval_set` trong `orders.csv`.

### 2.2. Schema từng file

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

### 3.1. Đơn hàng & eval_set

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số người dùng | **206,209** |
| Tổng số đơn hàng | **3,421,083** |
| Đơn hàng trung bình / user | **16.6** (median: 10, min: 4, max: 100) |

**Phân bố `eval_set`:**

| eval_set | Số lượng | % | Vai trò trong Global |
|----------|----------|---|----------------------|
| `prior` (lịch sử) | 3,214,874 | 94.0% | **Xây dựng model**: co-occurrence matrix, graph edges, TF-IDF |
| `train` (huấn luyện) | 131,209 | 3.8% | **Tune hyperparameters**: threshold Confidence, params KG, weights Hybrid |
| `test` (đánh giá) | 75,000 | 2.2% | **Đánh giá cuối**: báo cáo metrics (chỉ dùng 1 lần) |

### 3.2. Sản phẩm & Danh mục

| Chỉ số | Giá trị |
|--------|---------|
| Tổng sản phẩm | **49,688** |
| Số ngành hàng (department) | **21** |
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

## 4. Nhận xét & Hệ quả cho Global Recommender System

### 4.1. Dữ liệu đủ lớn cho Item-Oriented

- 32.4M interactions từ 3.2M đơn hàng cho phép tính **co-occurrence** và **PMI** chất lượng cao giữa 50K sản phẩm.
- Mỗi cặp sản phẩm có đủ dữ liệu để ước lượng xác suất mua kèm (P(A∩B)) một cách đáng tin cậy.
- 21 departments là nguồn phân loại sản phẩm ở mức logic (ngành hàng), phù hợp cho cả CB (TF-IDF) và KG (cấu trúc đồ thị).

### 4.2. Sparsity của Item-Item Matrix

- Ma trận co-occurrence item-item (50K×50K) có mật độ thưa: chỉ khoảng 1–2% cặp sản phẩm từng xuất hiện cùng nhau trong cùng đơn hàng.
- Ma trận co-occurrence dùng **Confidence unified scoring** (ochiai × confidence × log1p) để tính điểm mua kèm.
- Với những cặp không xuất hiện cùng nhau (99% ma trận), dùng **Content-Based similarity** (TF-IDF) để fallback.

### 4.3. Imbalance reorder

- ~59% reorder vs ~41% first-time purchase — thể hiện hành vi mua lặp lại phổ biến.
- Trong global Confidence, reorder không phải vấn đề vì co-occurrence đếm **mọi lần xuất hiện cùng nhau**, không phân biệt reorder hay không.
- Tuy nhiên, nếu muốn ưu tiên sản phẩm mới (first-time), có thể dùng Confidence có trọng số theo `reordered=0`.

### 4.4. Long-tail products

- Nhiều sản phẩm có tần suất xuất hiện thấp trong prior (long-tail).
- Với các sản phẩm long-tail, co-occurrence với sản phẩm khác rất thấp → Confidence không đủ tin cậy.
- **Content-Based filtering** (TF-IDF từ `product_name` + `department`) là giải pháp fallback cho long-tail items:
  - Dựa trên nội dung tên sản phẩm (text) thay vì interaction count
  - Không bị ảnh hưởng bởi sparsity của co-occurrence

### 4.5. Temporal patterns

- Dữ liệu có `order_dow`, `order_hour_of_day`, `days_since_prior_order` — các thông tin về thời gian.
- Trong global approach, temporal có thể dùng để:
  - **Weight co-occurrence**: ưu tiên các cặp sản phẩm xuất hiện cùng nhau trong cùng khung giờ / ngày
  - **Phân tích seasonal**: sản phẩm theo mùa (kem vào mùa hè, sô-cô-la nóng vào mùa đông)
- Tuy nhiên, ở phiên bản đầu, temporal chưa được đưa vào model chính.

### 4.6. Chia tập theo hướng Global

Khác với per-user (mỗi user có prior/train/test riêng), hướng Global dùng 3 tập như sau:

| Tập | Số đơn | Vai trò trong pipeline |
|-----|--------|------------------------|
| **prior** (100%) | 3,214,874 | **Xây dựng toàn bộ model** — không có "train" model theo nghĩa supervised learning. Toàn bộ co-occurrence matrix, graph edges, TF-IDF vectors đều được tính từ prior. |
| **train** (100%) | 131,209 | **Tune hyperparameters** — dùng ground truth từ train set để tìm threshold tối ưu cho Confidence (freq_min), restart_prob cho KG, và trọng số α/β cho Hybrid. |
| **test** (100%) | 75,000 | **Đánh giá cuối cùng** — chỉ chạy 1 lần duy nhất sau khi đã tune xong mọi tham số. Dùng metrics: Hit Rate@K, Precision@K, F1@K, NDCG@K, MAP@K. |

**Cơ chế đánh giá cụ thể:**
- Không phải "train model rồi predict test" như supervised learning thông thường
- Model (Confidence / KG) được xây từ prior và không thay đổi
- Với mỗi đơn trong test:
  1. Lấy danh sách sản phẩm trong đơn đó
  2. Với mỗi sản phẩm A, dùng Confidence/KG → top-N sản phẩm hay mua kèm với A
  3. So sánh top-N với các sản phẩm còn lại trong đơn (ground truth)
  4. Tính Hit Rate@K, Precision@K, F1@K, NDCG@K, MAP@K

### 4.7. Lưu ý kỹ thuật

- File products.csv chứa ký tự đặc biệt (em dash, accented characters) → cần đọc với encoding `utf-8`.
- Một số tên sản phẩm có chứa dấu phẩy trong ngoặc kép → cần dùng `csv.DictReader` thay vì split bằng dấu phẩy thủ công.
- File `order_products__prior.csv` có 32.4M records → cần xử lý theo chunk để tránh tràn bộ nhớ.
- Ma trận co-occurrence 50K×50K ở dạng dense là ~20GB → **bắt buộc dùng sparse matrix** (scipy.sparse).

## 5. Tóm tắt

| Khía cạnh | Đánh giá |
|-----------|----------|
| ✅ Kích thước | Phù hợp, đủ lớn để xây item-item relationships |
| ✅ Chất lượng | Dữ liệu sạch, có cấu trúc rõ ràng |
| ✅ Đa dạng | 50K sản phẩm, 21 departments |
| ✅ Temporal | Có timestamp, có thể dùng để phân tích mùa vụ |
| ✅ Split sẵn | prior / train / test — prior xây model, train tune params, test eval |
| ⚠️ Sparsity | Item-item co-occurrence thưa — cần Confidence + CB fallback |
| ⚠️ Long-tail | Sản phẩm ít xuất hiện → cần CB fallback dựa trên nội dung |

Dataset này hoàn toàn phù hợp để xây dựng hệ thống gợi ý sản phẩm **global (item-oriented)** với 4 models đã định nghĩa: **Content-Based (CB)**, **Collaborative Filtering (Confidence unified scoring)**, **Knowledge Graph (KG)**, và **Hybrid**.
