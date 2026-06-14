import os
import sys

# Thêm thư mục hiện tại vào path
sys.path.insert(0, ".")

from src.config import ENS_CB_THRESHOLD
from src.models.ensemble import EnsembleModel

print("--- Đang tải mô hình Ensemble... ---")
e = EnsembleModel.load()

# 1. Chạy recommend chính thức
recs = e.recommend(1, use_cb_filter=True)
print(f"\n[KẾT QUẢ] Ensemble+CB cho product 1: {len(recs)} recs")

# 2. Nếu không ra kết quả, tự động bóc tách từng bước để tìm lỗi
if not recs:
    print("\n[DEBUG] Phát hiện 0 kết quả. Bắt đầu phân tích nguyên nhân:")

    # Lấy danh sách ứng viên thô
    raw = e.get_raw_candidates(1, top_k=e.top_k)
    print(f"  -> Số lượng ứng viên thô (Raw candidates): {len(raw)}")

    if raw:
        # Thử đi qua bộ lọc CB
        filtered = e.cb_filter.filter(1, raw, threshold=ENS_CB_THRESHOLD)
        print(f"  -> Sau khi qua bộ lọc CB (Ngưỡng {ENS_CB_THRESHOLD}): {len(filtered)}")

        # Kiểm tra chi tiết 5 ứng viên đầu tiên
        print("\n  -> Chi tiết 5 ứng viên đầu tiên:")
        for pid, sc in raw[:5]:
            in_cb = pid in e.cb_filter.product_id_to_idx
            print(f"     + Sản phẩm ID {pid}: Score Ensemble = {sc:.4f} | Có trong ma trận CB? {in_cb}")
    else:
        print("  -> Tầng Ensemble gốc hoàn toàn không sinh ra ứng viên nào!")