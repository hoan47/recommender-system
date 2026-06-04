<!-- cSpell:disable -->
# QUY TẮC AN TOÀN

## 1. Nguyên tắc chung
- KHÔNG sửa logic code nếu không được yêu cầu
- LUÔN kiểm tra syntax trước và sau khi thay đổi

## 2. File dữ liệu
- KHÔNG commit/push các file dữ liệu gốc (csv, json, sql...) lên git
- Phải có **`.gitignore`** để loại trừ file dữ liệu trước khi commit
- Chỉ commit code, cấu hình, tài liệu — không commit data
