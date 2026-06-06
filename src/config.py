"""Cấu hình tập trung — đường dẫn + hằng số"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in [MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# CB
CB_MIN_DF = 5          # term phải xuất hiện >= 5 sản phẩm
CB_MAX_DF = 0.8        # term xuất hiện <= 80% sản phẩm
CB_MAX_FEATURES = 10000
TOP_K = 100            # giữ top-K tương tự

# SPMI
SPMI_K = 3             # shift threshold cho SPMI
TOTAL_PRIOR_ORDERS = 3214874
SPMI_TOP_K = 100       # giữ top-K mỗi dòng

# KG
KG_DIM = 64
KG_WALK_LENGTH = 20
KG_NUM_WALKS = 50
KG_EPOCHS = 1

# Hybrid
HYBRID_ALPHA = 0.6     # trọng số SPMI
HYBRID_BETA = 0.4      # trọng số KG
HYBRID_CB_THRESH = 0.8 # ngưỡng lọc substitute

# Đánh giá
EVAL_KS = (5, 10, 20)