# Giải thích các trường dữ liệu

Dữ liệu từ **Instacart Market Basket Analysis** (Kaggle).

---

## 1. `aisles.csv` — Danh mục lối đi

| Trường | Ý nghĩa |
|--------|---------|
| `aisle_id` | Mã lối đi |
| `aisle` | Tên lối đi (vd: `prepared soups salads`) |

## 2. `departments.csv` — Danh mục gian hàng

| Trường | Ý nghĩa |
|--------|---------|
| `department_id` | Mã gian hàng |
| `department` | Tên gian hàng (vd: `frozen`) |

## 3. `orders.csv` — Đơn hàng

| Trường | Ý nghĩa |
|--------|---------|
| `order_id` | Mã đơn hàng |
| `user_id` | Mã người dùng |
| `eval_set` | Nhãn tập: `train` (lịch sử + huấn luyện), `test` (dự đoán) |
| `order_number` | Đơn thứ mấy của user (1 = đầu tiên) |
| `order_dow` | Ngày trong tuần (0 = Chủ Nhật) |
| `order_hour_of_day` | Giờ đặt hàng (string, vd: `"08"`) |
| `days_since_prior_order` | Số ngày từ đơn trước; **rỗng `""`** nếu là đơn đầu |

## 4. `order_products__train.csv` / `order_products__test.csv` — Sản phẩm trong đơn

| Trường | Ý nghĩa |
|--------|---------|
| `order_id` | Mã đơn hàng |
| `product_id` | Mã sản phẩm |
| `add_to_cart_order` | Thứ tự bỏ vào giỏ (1 = đầu tiên) |
| `reordered` | `1` = đã mua trước đây, `0` = chưa |

## 5. `products.csv` — Sản phẩm

| Trường | Ý nghĩa |
|--------|---------|
| `product_id` | Mã sản phẩm |
| `product_name` | Tên sản phẩm |
| `aisle_id` | Mã lối đi → `aisles.aisle_id` |
| `department_id` | Mã gian hàng → `departments.department_id` |

## Quan hệ giữa các bảng

```
products.aisle_id    → aisles.aisle_id
products.department_id → departments.department_id
order_products.order_id   → orders.order_id
order_products.product_id → products.product_id
```

## Ghi chú

- `eval_set = train`: dùng để tính SPMI, `test`: dự đoán.
- `days_since_prior_order` ở đơn đầu là chuỗi rỗng `""`, không phải NaN.
- `order_hour_of_day` là string (`"08"`), cần ép kiểu khi xử lý.