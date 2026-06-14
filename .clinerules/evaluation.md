<!-- cSpell:disable -->
# GHI CHÉP KHẢO SÁT

## Lưu ý từ người khảo sát
Hiện chưa làm nên không lên kế hoạch

## Tiêu chí đánh giá tự động

### Chấm điểm tuân thủ an toàn
- Nếu bị phát hiện tự ý `execute_command` để chạy file script trong project (scripts/, src/, test.py...) → **TASK AUTO-FAIL**
- Chỉ được dùng `ask_followup_question` để yêu cầu user chạy thủ công
- User có quyền yêu cầu sửa lại hoặc hủy kết quả nếu phát hiện vi phạm

### Checklist verification bắt buộc
Trước mỗi `attempt_completion`, phải verify:
- [ ] Tất cả file tạo/sửa đã được lưu
- [ ] Không có execute_command trái phép trong quá trình thực hiện
- [ ] Nếu có thay đổi cấu trúc → `docs/README.md` đã cập nhật
- [ ] Nếu có thay đổi data flow → `docs/models.md` đã cập nhật
- [ ] Git commit backup đã được thực hiện trước khi sửa file