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

## 2. Pipeline model chính
- Dự án chỉ tập trung vào **pipeline model chính** tại `scripts/model/` (7 bước: 01→07)
- Các script khác ngoài `scripts/model/` không thuộc pipeline chính, không cập nhật trong docs
- Thứ tự chạy: `01_load_data` → `02_cb_filter` → `03_item_cf` → `04_item_cf_neural` → `05_kg_metapath` → `06_ensemble` → `07_eval_llm`

## 3. Thông số dữ liệu chính thức
| Chỉ số | Giá trị |
|--------|---------|
| Sản phẩm food (sau lọc) | **36,181** (giảm từ 49,688) |
| Records food (sau lọc) | **31,919,315** (giảm từ 33,819,108) |
| Non-food loại khỏi products | 13,507 (27.2%) |
| Non-food records loại khỏi data | 1,899,791 (6.0%) |
| Số đơn hàng sau lọc | 3,318,066 |

## 4. Kiểm tra dữ liệu
- Đường dẫn file dữ liệu: `data/` (KHÔNG commit lên git)
- File `aisles.csv` không được sử dụng trong dự án này
- KHÔNG có file `order_products__test.csv` — ground truth test nằm trong `order_products__train.csv`, phân biệt qua `orders.csv[eval_set]`
- File prior 32.4M records → xử lý chunk-based
- `products.parquet` (36,181 records) đã lọc non-food — tạo từ `model/01_load_data.py`
- `order_products.parquet` (31,919,315 records) đã lọc non-food — tạo từ `model/01_load_data.py`

## 5. Cập nhật docs khi thay đổi cấu trúc
- Khi thêm/xóa/di chuyển file hoặc thư mục: phải cập nhật `docs/README.md` (sơ đồ cấu trúc)
- Nếu thay đổi ảnh hưởng đến data flow/kiến trúc: cập nhật `docs/models.md`
- Chỉ pipeline model chính `scripts/model/` mới được đề cập trong docs chính

## 6. Hỏi user khi không chắc chắn
- Khi không chắc chắn về điều gì → hỏi lại user trước khi làm

## 7. Kiểm tra execute_command bắt buộc
TRƯỚC mỗi lần gọi `execute_command`, phải kiểm tra tất cả các mục sau:

- [ ] Command có gọi file `.py`, `.sh`, `.js`, `.bat` nằm trong project không? (VD: `python scripts/...`, `python .\scripts\...`, `./scripts/...`, `node src/...`)
- [ ] Nếu CÓ → KHÔNG được gọi `execute_command`, thay bằng `ask_followup_question` yêu cầu user tự chạy
- [ ] Nếu KHÔNG → chỉ được gọi cho: pip, python -c "code inline ngắn", git, mkdir, dir, cd, copy, del, echo, type, findstr, npm, docker, kubectl