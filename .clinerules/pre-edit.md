<!-- cSpell:disable -->
# KIỂM TRA TRƯỚC KHI CHỈNH SỬA

## 1. Git backup
- TRƯỚC khi sửa bất kỳ file nào: phải có git backup commit
- Nếu chưa có git → báo user khởi tạo git trước
- Commit message format: [type] ngắn gọn bằng tiếng Việt
  - `[backup]` cho commit backup trước khi chỉnh sửa
  - `[fix]` cho commit sửa lỗi
  - `[update]` cho commit cập nhật tài liệu/code
  - `[feat]` cho commit thêm tính năng mới

## 2. Đọc docs trước khi làm
- Trước khi implement: đọc docs/data_evaluation.md, docs/models.md, docs/implementation_plan.md
- Tuân theo thứ tự implement trong implementation_plan.md
- KHÔNG implement theo thứ tự khác nếu chưa có lý do chính đáng

## 3. Kiểm tra dữ liệu
- Đường dẫn file dữ liệu: `data/` (KHÔNG commit lên git)
- File `aisles.csv` không được sử dụng trong dự án này
- KHÔNG có file `order_products__test.csv` — ground truth test nằm trong `order_products__train.csv`, phân biệt qua `orders.csv[eval_set]`
- File prior 32.4M records → xử lý chunk-based

## 4. Hỏi user khi không chắc chắn
- Khi không chắc chắn về điều gì → hỏi lại user trước khi làm
