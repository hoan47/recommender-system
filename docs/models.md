# 🧠 Mô hình gợi ý sản phẩm — Recommender System

Đang dùng **4 models** được định nghĩa trong file:

1. **Content-Based (CB)**
   - Tìm sản phẩm có nội dung tương tự (TF-IDF vectors)
   - Mục đích: **lọc sản phẩm GIỐNG (substitute)**, không dùng để gợi ý chính
   - Chỉ là baseline để so sánh

2. **Collaborative Filtering (SPMI)**

3. **Knowledge Graph (RWR)**

4. **Hybrid**

**Điểm khác biệt chính:**
- **CB**: Dùng để **LOẠI** sản phẩm tương tự (substitute), không phải gợi ý chính
- **SPMI + KG**: Tìm complementary products (mua kèm) với ưu tiên cross-department
- **Hybrid**: Kết hợp cả hai phương pháp với CB làm filter