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

## 4. Danh sách command bị cấm (forbidden commands)
TRƯỚC mỗi lần gọi `execute_command`, phải kiểm tra command có chứa một trong các pattern sau KHÔNG.
Nếu CÓ → không được gọi `execute_command`, thay bằng `ask_followup_question` yêu cầu user tự chạy.

### Pattern file script trong project (bị cấm tuyệt đối):
- `python <đường_dẫn_file_trong_project>` — ví dụ: `python scripts/09_eval_cb_similarity.py`
- `python .\<đường_dẫn>` — ví dụ: `python .\scripts\01_load_data.py`
- `python3 <đường_dẫn_file_trong_project>`
- `./<đường_dẫn_file_trong_project>` — ví dụ: `./scripts/run.sh`
- Bất kỳ lệnh nào có path bắt đầu bằng `scripts/`, `src/`, `test.py`, `data/` và gọi interpreter (python, node, bash...)

### Pattern command hệ thống (được phép):
- `python -c "..."` (code ngắn inline, không gọi file project)
- `pip install ...`
- `git ...`
- `mkdir`, `dir`, `cd`, `copy`, `del`, `echo`, `type`, `findstr`
- `npm install`, `npm run ...` (nếu không gọi script project)
- `docker ...`, `kubectl ...`

### Quy trình thay thế khi cần chạy script project:
```markdown
Không tự chạy. Thay vào đó dùng `ask_followup_question`:
- Nêu rõ file cần chạy + lý do
- Đưa câu lệnh chính xác để user copy-paste
- Yêu cầu user gửi lại kết quả (stdout + stderr nếu có)