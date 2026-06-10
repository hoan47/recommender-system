"""
Cấu hình tập trung cho toàn bộ dự án.
Mọi hyperparameter, đường dẫn, hằng số đặt tại 1 file.
"""
import os

# ============================================================
# Đường dẫn thư mục
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# Files dữ liệu gốc
# ============================================================
ORDERS_FILE = os.path.join(DATA_DIR, "orders.csv")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products.csv")
ORDER_PRODUCTS_PRIOR = os.path.join(DATA_DIR, "order_products__prior.csv")
ORDER_PRODUCTS_TRAIN = os.path.join(DATA_DIR, "order_products__train.csv")
AISLES_FILE = os.path.join(DATA_DIR, "aisles.csv")
DEPARTMENTS_FILE = os.path.join(DATA_DIR, "departments.csv")

# ============================================================
# Files processed (lưu sau khi preprocess)
# ============================================================
PRODUCT_VECTORS_FILE = os.path.join(PROCESSED_DIR, "product_vectors.npz")

# ============================================================
# Hyperparameters — CB Filter
# ============================================================
CB_THRESHOLD = 0.8          # cosine similarity >= threshold → substitute → loại
CB_N_GRAM_RANGE = (1, 2)    # TF-IDF: unigram + bigram
CB_MAX_FEATURES = 15000      # TF-IDF: max số features từ tên sản phẩm

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
# Hyperparameters — Node2Vec
# ============================================================
N2V_EMBEDDING_DIM = 128
N2V_WALK_LENGTH = 25
N2V_NUM_WALKS = 20
N2V_P = 1.0
N2V_Q = 1.0
N2V_WORKERS = 4
N2V_EDGE_THRESHOLD = 10      # edge giữa 2 product nếu co-occur count >= threshold
N2V_TOP_K = 100

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
ENS_GAMMA = 0.25                # trọng số Node2Vec
ENS_TOP_K = 100                 # top-K sau ensemble (trước CB filter)
ENS_FINAL_K = 10                # top-K cuối cùng output

# ============================================================
# IO
# ============================================================
CHUNKSIZE = 500000              # chunksize cho đọc CSV lớn
RANDOM_SEED = 42                # seed cho reproducibility