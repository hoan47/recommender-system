<!-- cSpell:disable -->
# .clinerules Overview

## `rules/` — Rules (AI tự động tuân theo)
- Luật bắt buộc, không cần hỏi user
- Ví dụ: không sửa logic, kiểm tra syntax, nói tiếng Việt

## `hooks/` — Hooks (AI kiểm tra + hỏi user)
- Các bước kiểm tra trước khi hành động
- Ví dụ: git backup trước khi sửa, hỏi user nếu không chắc

## `skills/` — Kỹ năng
- Hướng dẫn AI cách làm 1 việc cụ thể (ví dụ: cách train model, cách deploy)

## `workflows/` — Quy trình làm việc
- Các bước tuần tự cho 1 tác vụ phức tạp (ví dụ: quy trình fix bug)