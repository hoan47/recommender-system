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
CB_N_GRAM_RANGE = (1, 2)    # TF-IDF: word 1-gram đến 2-gram (tên sản phẩm)
CB_MAX_FEATURES = 15000      # TF-IDF: max số features từ tên sản phẩm

# CB Multi-field — trọng số cho từng trường tiếng Việt
CB_NAME_WEIGHT = 1.0         # trọng số tên sản phẩm (quan trọng nhất)
CB_AISLE_WEIGHT = 0.8        # trọng số aisle (trung bình)
CB_DEPT_WEIGHT = 0.6         # trọng số department (tổng quát nhất)
CB_AISLE_N_GRAM_RANGE = (1, 1)   # aisle ngắn, chỉ unigram
CB_AISLE_MAX_FEATURES = 500       # 127 aisle × vài từ → 500 đủ
CB_DEPT_N_GRAM_RANGE = (1, 1)     # department ngắn, chỉ unigram
CB_DEPT_MAX_FEATURES = 100        # 20 dept × vài từ → 100 đủ

# Ensemble Count + TF-IDF
CB_ALPHA = 0.5                  # trọng số Count Vectorizer (TF-IDF weight = 1 - alpha)
CB_COUNT_N_GRAM_RANGE = (1, 1)  # Count Vectorizer: word unigram (đếm từ đơn)
CB_COUNT_MAX_FEATURES = 15000   # Count Vectorizer: max features

# ============================================================
# Hyperparameters — Item-CF (Item-Based Collaborative Filtering)
# Ochiai coefficient = Cosine similarity trên binary vector → Item-Based CF
# ============================================================
ITEMCF_MIN_SUPPORT = 10     # pair xuất hiện < 10 lần → bỏ qua
ITEMCF_TOP_K = 100          # số candidate giữ lại cho mỗi product (trước ensemble)

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
# Hyperparameters — Metapath2Vec (IKG embedding, Metapath Walk)
# Xây dựng Đồ thị Tri thức đa thể với 2 kịch bản Metapath Walk:
#   1. Behavioral: P --CO_OCCUR--> P --CO_OCCUR--> P
#   2. Semantic:   P --BELONGS_TO--> A --random--> P
# ============================================================
MW_EMBEDDING_DIM = 128
MW_WALK_LENGTH = 25
MW_NUM_WALKS = 20
MW_WORKERS = 4
MW_WINDOW = 10
MW_NEGATIVE = 10
MW_EPOCHS = 20
MW_EDGE_THRESHOLD = 10
MW_TOP_K = 100
MW_METAPATH_BEHAVIORAL_RATIO = 0.5   # 50% Behavioral, 50% Semantic

# ============================================================
# Hyperparameters — Ensemble
# ============================================================
ENS_ALPHA = 0.5                 # trọng số Item-CF (Ochiai)
ENS_BETA = 0.25                 # trọng số Item2Vec
ENS_GAMMA = 0.25                # trọng số Metapath2Vec (IKG embedding)
ENS_TOP_K = 100                 # top-K sau ensemble (trước CB filter)
ENS_FINAL_K = 10                # top-K cuối cùng output
ENS_CB_THRESHOLD = 0.25          # ngưỡng CB filter khi dùng hybrid ensemble

# ============================================================
# Product Filter Strategy
# Lọc non-food products khỏi train data để model chỉ học
# các pattern mua kèm thực phẩm (grocery).
# Dùng department name thay vì ID để dễ đọc & đồng bộ với survey.
# ============================================================
EXCLUDED_DEPARTMENT_NAMES = ['other', 'pets', 'personal care', 'household', 'babies', 'missing']
# other — không phân loại rõ ràng
# pets — thức ăn, phụ kiện thú cưng
# personal care — sữa tắm, dầu gội, kem đánh răng, mỹ phẩm
# household — đồ gia dụng, bột giặt, giấy vệ sinh, túi nilon
# babies — tã bỉm, khăn ướt, đồ chơi, phụ kiện em bé, kể cả baby food formula
# missing — dữ liệu lỗi/thiếu thông tin ngành hàng

RANDOM_SEED = 42