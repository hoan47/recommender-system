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
- Trước khi implement: đọc docs/data_evaluation.md, docs/models.md
- KHÔNG implement theo thứ tự khác nếu chưa có lý do chính đáng

## 3. Kiểm tra dữ liệu
- Đường dẫn file dữ liệu: `data/` (KHÔNG commit lên git)
- File `aisles.csv` không được sử dụng trong dự án này
- KHÔNG có file `order_products__test.csv` — ground truth test nằm trong `order_products__train.csv`, phân biệt qua `orders.csv[eval_set]`
- File prior 32.4M records → xử lý chunk-based

## 4. Cập nhật docs khi thay đổi cấu trúc
- Khi thêm/xóa/di chuyển file hoặc thư mục: phải cập nhật `docs/README.md` (sơ đồ cấu trúc)
- Nếu thay đổi ảnh hưởng đến data flow/kiến trúc: cập nhật `docs/models.md`

## 5. Hỏi user khi không chắc chắn
- Khi không chắc chắn về điều gì → hỏi lại user trước khi làm

## 6. Kiểm tra execute_command bắt buộc
TRƯỚC mỗi lần gọi `execute_command`, phải kiểm tra tất cả các mục sau:

- [ ] Command có gọi file `.py`, `.sh`, `.js`, `.bat` nằm trong project không? (VD: `python scripts/...`, `python .\scripts\...`, `./scripts/...`, `node src/...`)
- [ ] Nếu CÓ → KHÔNG được gọi `execute_command`, thay bằng `ask_followup_question` yêu cầu user tự chạy
- [ ] Nếu KHÔNG → chỉ được gọi cho: pip, python -c "code inline ngắn", git, mkdir, dir, cd, copy, del, echo, type, findstr, npm, docker, kubectl