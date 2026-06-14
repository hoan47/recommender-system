"""
Cấu hình tập trung cho toàn bộ dự án.
Mọi hyperparameter, đường dẫn, hằng số đặt tại 1 file.
"""
import os

# ============================================================
# Đường dẫn thư mục
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# Files dữ liệu gốc
# ============================================================
ORDERS_FILE = os.path.join(PROJECT_ROOT, "data/orders.csv")
PRODUCTS_FILE = os.path.join(PROJECT_ROOT, "data/products.csv")
ORDER_PRODUCTS_PRIOR = os.path.join(PROJECT_ROOT, "data/order_products__prior.csv")
ORDER_PRODUCTS_TRAIN = os.path.join(PROJECT_ROOT, "data/order_products__train.csv")
AISLES_FILE = os.path.join(PROJECT_ROOT, "data/aisles.csv")
DEPARTMENTS_FILE = os.path.join(PROJECT_ROOT, "data/departments.csv")

# ============================================================
# Xử lý dữ liệu (chunk-based)
# ============================================================
CHUNKSIZE = 500000  # số records mỗi chunk khi đọc file prior 32.4M records

# ============================================================
# Hyperparameters — CB (Content-Based Vectorizer)
# ============================================================
CB_N_GRAM_RANGE = (1, 2)    # TF-IDF: word 1-gram đến 2-gram
CB_MAX_FEATURES = 15000      # TF-IDF: max số features từ tên sản phẩm

# Ensemble Count + TF-IDF
CB_ALPHA = 0.8               # trọng số Count Vectorizer (TF-IDF weight = 1 - alpha)
CB_COUNT_N_GRAM_RANGE = (1, 1)  # Count Vectorizer: word unigram (đếm từ đơn)
CB_COUNT_MAX_FEATURES = 15000   # Count Vectorizer: max features

# ============================================================
# Hyperparameters — Ochiai
# ============================================================
OCHIAI_MIN_SUPPORT = 10     # pair xuất hiện < 30 lần → bỏ qua
OCHIAI_TOP_K = 100          # số candidate giữ lại cho mỗi product (trước ensemble)

# ============================================================
# Hyperparameters — Item2Vec
# ============================================================
I2V_VECTOR_SIZE = 128
I2V_WINDOW = 5
I2V_MIN_COUNT = 10
I2V_NEGATIVE = 10
I2V_EPOCHS = 20
I2V_WORKERS = 4
I2V_TOP_K = 100

# ============================================================
# Hyperparameters — DeepWalk (graph embedding, uniform random walk)
# ============================================================
DW_EMBEDDING_DIM = 128
DW_WALK_LENGTH = 25
DW_NUM_WALKS = 20
DW_WORKERS = 4
DW_WINDOW = 10
DW_NEGATIVE = 10
DW_EPOCHS = 20
DW_EDGE_THRESHOLD = 10
DW_TOP_K = 100

# ============================================================
# Hyperparameters — Association Rules (từ co-occurrence matrix)
# ============================================================
ARM_MIN_SUPPORT = 0.0001        # support threshold (tỷ lệ)
ARM_MIN_CONFIDENCE = 0.1        # confidence threshold
ARM_MIN_LIFT = 1.5              # lift threshold
ARM_TOP_K = 100

# ============================================================
# Hyperparameters — Ensemble
# ============================================================
ENS_ALPHA = 0.5                 # trọng số Ochiai
ENS_BETA = 0.25                 # trọng số Item2Vec
ENS_GAMMA = 0.25                # trọng số DeepWalk (graph embedding)
ENS_TOP_K = 100                 # top-K sau ensemble (trước CB filter)
ENS_FINAL_K = 10                # top-K cuối cùng output
ENS_CB_THRESHOLD = 0.25          # ngưỡng CB filter khi dùng hybrid ensemble
CB_THRESHOLD = 0.25              # ngưỡng CB similarity để phân loại Substitute (>=) vs Complementary

# ============================================================
# Product Filter Strategy
# Lọc non-food products khỏi train data để model chỉ học
# các pattern mua kèm thực phẩm (grocery).
# ============================================================
EXCLUDED_DEPARTMENTS = [8, 11, 17, 2, 21]
# pets (8), personal care (11), household (17), other (2), missing (21)

EXCLUDED_AISLES = [82, 102]
# baby accessories (82), baby bath body care (102)
# Giữ aisle 92 (baby food formula) — thực phẩm cho trẻ

RANDOM_SEED = 42