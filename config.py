import os

DATA_DIR         = r"C:\Users\b2h16\OneDrive\Máy tính\recommendation-system\data"
PATH_TRAIN       = os.path.join(DATA_DIR, "order_products_train.csv")
PATH_PRODUCTS    = os.path.join(DATA_DIR, "products.csv")
PATH_DEPARTMENTS = os.path.join(DATA_DIR, "departments.csv")
PATH_OUTPUT_CSV  = os.path.join(DATA_DIR, "evaluation_results.csv")
PATH_DEPT_DIR    = os.path.join(DATA_DIR, "department_direction.csv")

RAM_SCALE = 1.0
MIN_FREQ  = 50  # Số lần mua ít nhất của một sản phẩm để được coi là quan trọng
MAX_CASES = int(RAM_SCALE * 2000)

MIN_CONF        = 30.0
MIN_LIFT        = 1.05
ASYMMETRY_RATIO = 1.15

CB_MAX_DF      = 0.90
CB_MIN_DF      = 2
CB_FILTER_HIGH = 0.80

KG_SPMI_SHIFT    = 5
KG_TOP_SPMI_EDGE = 50
KG_RWR_STEPS     = 4
KG_RWR_RESTART   = 0.10
KG_RWR_WALKS     = 1000
KG_CO_MIN_COUNT  = 3

HYB_ALPHA = 0.85
