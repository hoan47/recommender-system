"""
Cấu hình tập trung cho toàn bộ dự án
Chứa đường dẫn thư mục và hằng số cho từng model
"""

from pathlib import Path

# Đường dẫn thư mục gốc, data, models, results
# PROJECT_ROOT tự động tính từ vị trí file này (src/config.py → lên 2 cấp)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"       # Chứa file CSV gốc (đã .gitignore)
MODELS_DIR = PROJECT_ROOT / "models"    # Chứa output model (đã .gitignore)
RESULTS_DIR = PROJECT_ROOT / "results"  # Chứa kết quả evaluation (đã .gitignore)

# Tự động tạo thư mục models/ và results/ nếu chưa có
for _d in [MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ===== Content-Based (CB) =====
# CB_MIN_DF: Term phải xuất hiện ít nhất trong N sản phẩm (loại bỏ term quá hiếm)
CB_MIN_DF = 5
# CB_MAX_DF: Term xuất hiện trong tối đa N% sản phẩm (loại bỏ term quá phổ biến)
CB_MAX_DF = 0.8
# CB_MAX_FEATURES: Số lượng term tối đa trong vocabulary (kiểm soát bộ nhớ)
CB_MAX_FEATURES = 10000
# TOP_K: Chỉ giữ K sản phẩm tương tự nhất mỗi dòng trong ma trận similarity
TOP_K = 100

# ===== Confidence (Item-based Collaborative Filtering) =====
# CONF_FREQ_MIN: Sản phẩm phải xuất hiện trong ít nhất N đơn mới được recommend
# Loại bỏ nhiễu từ sản phẩm quá hiếm (ví dụ: 2-5 đơn)
CONF_FREQ_MIN = 30
# CONF_TOP_K: Chỉ giữ K sản phẩm mua kèm mạnh nhất mỗi sản phẩm
CONF_TOP_K = 100

# ===== SPMI (Collaborative Filtering, deprecated) =====
# SPMI_K: Threshold shift cho SPMI (càng cao càng loại bỏ nhiều edges yếu)
SPMI_K = 10
# TOTAL_PRIOR_ORDERS: Tổng số đơn hàng trong prior (dùng cho công thức PMI)
TOTAL_PRIOR_ORDERS = 3214874
# SPMI_TOP_K: Chỉ giữ K sản phẩm mua kèm mạnh nhất mỗi sản phẩm
SPMI_TOP_K = 100

# ===== Knowledge Graph (KG) =====
# KG_DIM: Kích thước embedding vector cho mỗi node (product + department)
KG_DIM = 64
# KG_WALK_LENGTH: Độ dài mỗi random walk trên đồ thị
KG_WALK_LENGTH = 20
# KG_NUM_WALKS: Số lượng random walk cho mỗi node
KG_NUM_WALKS = 50
# KG_EPOCHS: Số epoch huấn luyện skip-gram (1 là đủ vì dữ liệu lớn)
KG_EPOCHS = 1

# ===== Hybrid =====
# HYBRID_ALPHA: Trọng số cho SPMI score (0.0 ~ 1.0)
# SPMI recall ~1-4% (yếu) → trọng số thấp
HYBRID_ALPHA = 0.2
# HYBRID_BETA: Trọng số cho KG score (0.0 ~ 1.0)
# KG recall ~11-25% (mạnh) → trọng số cao
HYBRID_BETA = 0.8
# HYBRID_CB_THRESH: Ngưỡng CB similarity để loại sản phẩm substitute
# Nếu CB_sim(A,B) > threshold → A và B quá giống → loại khỏi gợi ý
# 0.85: chỉ loại substitute rất giống (cùng tên gần như identical), giữ lại complementary
HYBRID_CB_THRESH = 0.85

# ===== Evaluation =====
# EVAL_KS: Các giá trị K để đánh giá recall@K
EVAL_KS = (5, 10, 20)