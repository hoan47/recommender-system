<!-- cSpell:disable -->
# QUY TẮC AN TOÀN

## 1. Nguyên tắc chung
- KHÔNG sửa logic code nếu không được yêu cầu
- LUÔN kiểm tra syntax trước và sau khi thay đổi
- TUYỆT ĐỐI KHÔNG tự chạy file script nằm trong project
- Nếu cần chạy file trong project → nhắn user tự chạy và gửi lại kết quả, không tự ý chạy bằng command


## 2. File dữ liệu
- KHÔNG commit/push các file dữ liệu gốc (csv, json, sql...) lên git
- Phải có **`.gitignore`** để loại trừ file dữ liệu trước khi commit
- Chỉ commit code, cấu hình, tài liệu — không commit data

## 3. Lưu ý dự án này
- Ma trận 50K×50K dense = ~20GB → **bắt buộc dùng scipy.sparse**
- `order_products__prior.csv` 32.4M records → xử lý chunk (chunksize=500K)
- Đọc CSV: luôn dùng `encoding='utf-8'` và `csv.DictReader` (có dấu phẩy trong ngoặc kép)
- Models output lưu trong `models/`, results trong `results/` — cả 2 đã .gitignore
