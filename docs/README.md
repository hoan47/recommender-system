# Instacart Bundle Recommendation System

## Tổng Quan Dự Án

Dự án xây dựng **Hệ thống Gợi ý Mua kèm (Bundle Recommendation System)** dựa trên tập dữ liệu Instacart Market Basket Analysis.

## Cấu Trúc Thư Mục Dự Án

```
recommender-system/
├── data/                    # Dữ liệu gốc (CSV) — không commit
│   ├── aisles.csv
│   ├── departments.csv
│   ├── order_products__prior.csv   # 32.4M records
│   ├── order_products__train.csv   # 1.38M records
│   ├── orders.csv                  # 3.42M records
│   ├── products.csv                # 49K records
│   ├── processed/                  # Dữ liệu đã xử lý (parquet) — không commit
│   └── survey/                     # Survey samples — không commit
├── docs/                    # Tài liệu dự án
│   ├── README.md            # File này — tổng quan & cấu trúc
│   ├── data_survey.md       # Khảo sát & thống kê dữ liệu
│   ├── models.md            # Mục đích dự án & phương pháp model
├── models/                  # Model output — không commit
├── results/                 # Kết quả — không commit
├── scripts/                 # Script chạy từng bước riêng lẻ
│   ├── 01_load_data.py      # Load & cache dữ liệu
│   ├── 02_cb_filter.py      # Train CB Filter
│   ├── 03_ochiai.py         # Train Ochiai + Confidence
│   ├── 04_item2vec.py       # Train Item2Vec
│   ├── 05_deepwalk.py       # Train DeepWalk
│   ├── 06_assoc_rules.py    # Train Association Rules
│   ├── 07_ensemble.py       # Ensemble + test
│   ├── 08_streamlit_app.py  # Streamlit Dashboard UI
│   ├── 09_eval_cb_distribution.py  # Đánh giá phân bố CB similarity → chọn ENS_CB_THRESHOLD
├── src/                     # Source code
│   ├── config.py            # Cấu hình tập trung
│   ├── evaluation/          # (dự trữ cho đánh giá sau)
│   ├── features/            # Xử lý dữ liệu & vector hóa
│   │   ├── loader.py
│   │   ├── product_filter.py   # Bộ lọc non-food departments/aisles
│   │   └── vectorizer.py
│   ├── utils/               # Hàm tiện ích dùng chung
│   │   ├── __init__.py
│   │   └── _numba_ops.py    # Numba-accelerated operations (co-occurrence counting, adjacency CSR)
│   └── models/              # Các model recommendation
│       ├── __init__.py
│       ├── assoc_rules.py
│       ├── cb_filter.py
│       ├── deepwalk.py
│       ├── ensemble.py
│       ├── item2vec.py
│       └── ochiai.py
└── .gitignore               # Loại trừ data/, models/, results/
```

## Tài Liệu Liên Quan

| File | Mô tả |
|------|-------|
| `docs/data_survey.md` | Khảo sát chi tiết tập dữ liệu Instacart |
| `docs/models.md` | Mục đích dự án, phân biệt bài toán, chiến lược dữ liệu & phương pháp |

## Cách chạy

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Cài dependencies (làm 1 lần)
pip install -r requirements.txt

# Chạy từng bước riêng lẻ (không cần chạy lại nếu đã train)
python scripts/01_load_data.py
python scripts/02_cb_filter.py
python scripts/03_ochiai.py
python scripts/04_item2vec.py
python scripts/05_deepwalk.py
python scripts/06_assoc_rules.py
python scripts/07_ensemble.py

# Chạy toàn bộ bằng cmd ngoài không dùng vscode đỡ nóng máy (nhớ xóa dữ liệu trong models)
cd C:\Users\b2h16\OneDrive\Máy tính\recommender-system

# Kích hoạt môi trường ảo (nếu có)
.venv\Scripts\activate

# Bước 1: Load data (nếu chưa làm)
python scripts/01_load_data.py

# Bước 2: Train CB Filter (nếu chưa làm)
python scripts/02_cb_filter.py

# Bước 3: Train Ochiai (nếu chưa làm)
python scripts/03_ochiai.py

# Bước 4: Train Item2Vec (~5-10 phút)
python scripts/04_item2vec.py

# Bước 5: Train DeepWalk (~3-5 phút)
python scripts/05_deepwalk.py

# Bước 6: Train Association Rules (~2-5 phút)
python scripts/06_assoc_rules.py

# Bước 7: Test Ensemble + CB Filter
python scripts/07_ensemble.py

# Bước 8: Streamlit Dashboard
streamlit run scripts/08_streamlit_app.py

# Bước 9: Đánh giá phân bố CB similarity (chọn ENS_CB_THRESHOLD)
python scripts/09_eval_cb_distribution.py
```

## Lưu Ý

- Dữ liệu gốc (CSV) nằm trong `data/` và **không được commit** lên git
- `models/` và `results/` cũng được loại trừ qua `.gitignore`
- Chi tiết về dữ liệu xem tại [data_survey.md](data_survey.md)
- Chi tiết về mục đích dự án & model xem tại [models.md](models.md)

---