import os
import pandas as pd

# 1. Định nghĩa đường dẫn file gốc và file đầu ra
input_path = r"C:\Users\b2h16\Downloads\products.csv"
output_path = (
    r"C:\Users\b2h16\OneDrive\Máy tính\recommender-system\data\processed\products_vi.csv"
)

# Tự động tạo thư mục processed nếu chưa có
os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    # 2. Đọc file csv và CHỈ LẤY đúng 2 cột bạn yêu cầu
    # Việc chỉ định usecols ở đây sẽ tự động loại bỏ (xóa) tất cả các cột còn lại
    df = pd.read_csv(input_path, usecols=["product_id", "product_name_vi"])

    # 3. Kiểm tra lại cấu trúc dữ liệu để chắc chắn chỉ còn 2 cột
    print("--- Xem trước 5 dòng dữ liệu đầu tiên ---")
    print(df.head())
    print(f"\nTổng số lượng sản phẩm: {len(df)}")

    # 4. Xuất ra file csv mới vào thư mục processed
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[Thành công] Đã lọc và lưu file mới tại: {output_path}")

except FileNotFoundError:
    print(f"[Lỗi] Không tìm thấy file tại đường dẫn: {input_path}")
except KeyError:
    print(
        "[Lỗi] Không tìm thấy cột 'product_id' hoặc 'product_name_vi' trong file gốc. Bạn vui lòng kiểm tra lại tiêu đề cột nhé."
    )
except Exception as e:
    print(f"[Lỗi hệ thống]: {e}")