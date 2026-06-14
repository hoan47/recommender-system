import os
import re
from collections import Counter
import pandas as pd

# Đường dẫn file của bạn
file_path = r"C:\Users\b2h16\OneDrive\Máy tính\recommender-system\data\processed\products_vi.csv"

try:
    print("=== ĐANG QUÉT TỰ ĐỘNG: PHÁT HIỆN TẤT CẢ CÁC CỤM CHỨA SỐ ===")
    df = pd.read_csv(file_path)
    products = df["product_name_vi"].dropna().astype(str).tolist()
    print(f"Tổng số dòng quét qua: {len(products):,}\n")

    discovered_patterns = []
    examples = {}

    # 1. Regex tìm "Số đứng trước" (Bao gồm cả số thập phân, phần trăm, kí tự đặc biệt liền sau)
    # Ví dụ: 500ml, 5 lit, 7 Inch, 2%, 1.5L
    pattern_num_first = r"\b\d+(?:[\.,]\d+)?\s*[%a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễđìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]+\b"

    # 2. Regex tìm "Số đứng sau" (Chữ/Kí tự đặc biệt đứng trước, số đứng sau)
    # Ví dụ: - 4 CT, pack 6, vỉ 4, x12, no.1
    pattern_num_last = r"\b[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễđìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ-]+\s*\d+\b"

    for p in products:
        # Tìm tất cả các cụm khớp theo 2 quy luật
        matches_first = re.findall(pattern_num_first, p, flags=re.IGNORECASE)
        matches_last = re.findall(pattern_num_last, p, flags=re.IGNORECASE)
        
        total_matches = matches_first + matches_last
        
        for m in total_matches:
            m_clean = m.strip()
            # Loại bỏ các số thuần túy (nếu có lọt vào)
            if not m_clean.isdigit():
                # Chuẩn hóa khoảng trắng để gom nhóm chính xác hơn
                m_norm = re.sub(r'\s+', ' ', m_clean).lower()
                discovered_patterns.append(m_norm)
                if m_norm not in examples:
                    examples[m_norm] = p

    # Thống kê tần suất
    pattern_counts = Counter(discovered_patterns)

    print("--- TOP 50 CỤM CHỨA SỐ TỰ ĐỘNG PHÁT HIỆN ĐƯỢC ---")
    print("Dưới đây là danh sách các hạt sạn (dung tích, quy cách) cần dọn dẹp:")
    print("-" * 80)
    
    for pattern, count in pattern_counts.most_common(50):
        print(f" ❌ Cụm tìm thấy: '{pattern}' | Xuất hiện: {count:,} lần")
        print(f"    Ví dụ gốc    : '{examples[pattern]}'\n")

    # Xuất file báo cáo chi tiết để bạn kiểm tra toàn bộ
    df_report = pd.DataFrame(pattern_counts.most_common(), columns=["Cum_Chua_So", "So_Lan_Xuat_Hien"])
    df_report["Vi_Du_Thuc_Te"] = df_report["Cum_Chua_So"].map(examples)
    
    output_name = "ket_qua_quet_so.csv"
    df_report.to_csv(output_name, index=False, encoding="utf-8-sig")
    
    print("-" * 80)
    print(f"Đã lưu toàn bộ danh sách quét được vào file: {os.path.abspath(output_name)}")
    print("Bạn chạy đoạn này rồi thảy kết quả hiển thị lên đây nhé, mình sẽ viết hàm RegEx xóa sạch tụi nó một lần luôn!")

except Exception as e:
    print(f"Gặp lỗi khi quét dữ liệu: {e}")