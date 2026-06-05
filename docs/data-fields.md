# Giải thích các trường dữ liệu

Dữ liệu từ **Instacart Market Basket Analysis** (Kaggle).  
Đã xem dữ liệu mẫu (5 dòng đầu) từng file để giải thích.

---

## 1. `aisles.csv` — Danh mục lối đi

Dữ liệu mẫu:

```
1 | prepared soups salads
2 | specialty cheeses
3 | energy granola bars
4 | instant foods
5 | marinades meat preparation
```

| Trường | Ý nghĩa |
|--------|---------|
| `aisle` | Tên lối đi (vd: `prepared soups salads`) |

---

## 2. `departments.csv` — Danh mục gian hàng

Dữ liệu mẫu:

```
1 | frozen
2 | other
3 | bakery
4 | produce
5 | alcohol
```

| Trường | Ý nghĩa |
|--------|---------|
| `department` | Tên gian hàng (vd: `frozen`) |

---

## 3. `orders.csv` — Đơn hàng

Dữ liệu mẫu:

| order_id | user_id | eval_set | order_number | order_dow | order_hour_of_day | days_since_prior_order |
|----------|---------|----------|------------|----------|------------------|----------------------|
| 2539329 | 1 | prior | 1 | 2 | 08 | *(rỗng)* |
| 2398795 | 1 | prior | 2 | 3 | 07 | 15.0 |
| 473747 | 1 | prior | 3 | 3 | 12 | 21.0 |

| Trường | Ý nghĩa |
|--------|---------|
| `order_id` | Mã đơn hàng (vd: `2539329`) |
| `user_id` | Mã người dùng (vd: `1`) |
| `eval_set` | Nhãn tập: `prior` (lịch sử) / `train` / `test` |
| `order_number` | Đơn thứ mấy của user (1 = đầu tiên) |
| `order_dow` | Ngày trong tuần (2, 3, 4,...) |
| `order_hour_of_day` | Giờ đặt hàng (vd: `08`, `07`, `12`) — lưu dạng string |
| `days_since_prior_order` | Số ngày từ đơn trước; **rỗng** nếu là đơn đầu |

---

## 4. `order_products__train.csv` — Sản phẩm trong đơn (train)

Dữ liệu mẫu:

| order_id | product_id | add_to_cart_order | reordered |
|----------|-----------|-----------------|-----------|
| 2 | 33120 | 1 | 1 |
| 2 | 28985 | 2 | 1 |
| 2 | 9327 | 3 | 0 |

| Trường | Ý nghĩa |
|--------|---------|
| `order_id` | Mã đơn hàng |
| `product_id` | Mã sản phẩm |
| `add_to_cart_order` | Thứ tự bỏ vào giỏ (1 = đầu tiên) |
| `reordered` | `1` = đã mua trước đây, `0` = chưa |

---

## 5. `order_products__test.csv` — Sản phẩm trong đơn (test)

Cấu trúc giống hệt `train`.

---

## 6. `products.csv` — Sản phẩm

Dữ liệu mẫu:

| product_id | product_name | aisle_id | department_id |
|-----------|-------------|---------|--------------|
| 1 | Chocolate Sandwich Cookies | 61 | 19 |
| 2 | All-Seasons Salt | 104 | 13 |

| Trường | Ý nghĩa |
|--------|---------|
| `product_id` | Mã sản phẩm |
| `product_name` | Tên sản phẩm |
| `aisle_id` | Mã lối đi |
| `department_id` | Mã gian hàng |

---

## Quan hệ giữa các bảng

```
products.aisle_id → aisles.aisle_id
products.department_id → departments.department_id
order_products.order_id → orders.order_id
order_products.product_id → products.product_id
```

## Ghi chú

- `eval_set = prior`: dùng làm dữ liệu lịch sử (tính SPMI).
- `eval_set = train`: đầu vào huấn luyện model.
- `eval_set = test`: dùng dự đoán, giữ nguyên cấu trúc train.
- `days_since_prior_order` ở đơn đầu là **rỗng `""`**, không phải NaN.
- `order_hour_of_day` trong file gốc là **string** (`"08"`), cần ép kiểu.