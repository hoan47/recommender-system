# config.py
# Tap trung tat ca duong dan va hang so cau hinh

import os

# ── Duong dan ─────────────────────────────────────────────────────────────────
DATA_DIR         = "data"
PATH_TRAIN       = os.path.join(DATA_DIR, "order_products_train.csv")
PATH_TEST        = os.path.join(DATA_DIR, "order_products_test.csv")
PATH_PRODUCTS    = os.path.join(DATA_DIR, "products.csv")
PATH_DEPARTMENTS = os.path.join(DATA_DIR, "departments.csv")
PATH_OUTPUT_CSV  = os.path.join(DATA_DIR, "evaluation_results.csv")
PATH_DEPT_DIR    = os.path.join(DATA_DIR, "department_direction.csv")

# ── Data loading ──────────────────────────────────────────────────────────────
RAM_SCALE = 3.0
# Giam min_freq tu 50 -> 20 de mo rong khong gian san pham goi y
# 50k SP voi 3M don -> SP mua >= 50 lan con ~26k SP (qua it)
# Giam xuong 20 -> ~40k SP, tang kha nang hit trong evaluation
MIN_FREQ  = 15
MAX_CASES = int(RAM_SCALE * 1000) 

# ── Department direction ──────────────────────────────────────────────────────
MIN_CONF        = 30.0
MIN_LIFT        = 1.05
ASYMMETRY_RATIO = 1.15

# ── Content-Based ─────────────────────────────────────────────────────────────
CB_MAX_DF      = 0.50
CB_MIN_DF      = 2
CB_FILTER_HIGH = 0.80   # cosine sim > nguong nay -> loai (substitute)
CB_FILTER_LOW  = 0.05

# ── Knowledge Graph ──────────────────────────────────────────────────────────
KG_TOP_EDGE_LIMIT = 50
KG_TOP_I2V_EDGE  = 20
KG_I2V_SIM_MIN   = 0.30
# Tang n_walks tu 300->1000 de RWR on dinh hon o top-10
KG_RWR_STEPS     = 6
KG_RWR_RESTART   = 0.15
KG_RWR_WALKS     = 1000
KG_MAX_BASKET    = 100
KG_CO_MIN_COUNT  = 3

# ── Hybrid ────────────────────────────────────────────────────────────────────
HYB_ALPHA = 0.20  # 20% CF, 80% KG (vi KG manh hon nhieu)