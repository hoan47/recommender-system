# Instacart Bundle Recommendation System

## Tổng Quan Dự Án

Dự án xây dựng **Hệ thống Gợi Ý Mua kèm (Bundle Recommendation System)** dựa trên tập dữ liệu Instacart Market Basket Analysis.

## Pipeline Model Chính (scripts/model/)

Dự án chỉ tập trung vào **7 bước pipeline model** chính, đặt trong `scripts/model/`:

```
recommender-system/
├── scripts/
│   └── model/                 # Pipeline model chính (1→7)
│       ├── 01_load_data.py    # Load & lọc dữ liệu (loại non-food)
│       ├── 02_cb_filter.py    # Content-Based Diversity Filter
│       ├── 03_item_cf.py      # Item-Based Collaborative Filtering (Ochiai + Confidence)
│       ├── 04_item_cf_neural.py # Item2Vec (Neural Item-Based CF)
│       ├── 05_kg_metapath.py  # KG Metapath embedding
│       ├── 06_ensemble.py     # Ensemble + CB Filter
│       └── 07_eval_llm.py     # LLM Evaluation
├── src/                       # Source code — thư viện Python
│   ├── __init__.py            # (rỗng)
│   ├── config.py              # Cấu hình tập trung (hyperparameters, đường dẫn)
│   ├── features/              # Xử lý dữ liệu & vector hóa
│   │   ├── __init__.py        # (rỗng)
│   │   ├── loader.py          # Đọc & merge dữ liệu từ CSV gốc
│   │   ├── product_filter.py  # Bộ lọc non-food departments/aisles
│   │   └── vectorizer.py      # TF-IDF, Count Vectorizer, Multi-field vector hóa
│   ├── models/                # Các model recommendation (class-based)
│   │   ├── cb_filter.py       # CBFilter — Content-Based Diversity Filter
│   │   ├── item_cf.py         # ItemCFModel — Item-Based CF (Ochiai + Confidence)
│   │   ├── item_cf_neural.py  # ItemCFNeuralModel — Item2Vec (Neural CF)
│   │   ├── kg_metapath.py     # KGMetapathModel — KG embedding + Metapath Walk
│   │   └── ensemble.py        # EnsembleModel — Weighted ensemble + CB Filter
│   └── utils/                 # Hàm tiện ích dùng chung
│       ├── __init__.py
│       └── _numba_ops.py      # Numba-accelerated operations (JIT-compiled)
├── data/                      # Dữ liệu gốc (CSV) — không commit
│   └── processed/             # Dữ liệu đã xử lý (parquet, CSV dịch) — không commit
├── models/                    # Model output — không commit
├── results/                   # Kết quả — không commit
├── docs/                      # Tài liệu dự án
│   ├── README.md              # File này — tổng quan & cấu trúc
│   ├── data_survey.md         # Khảo sát & thống kê dữ liệu
│   └── models.md              # Mục đích dự án & phương pháp model (chi tiết thuật toán)
├── .gitignore
├── requirements.txt
└── vietnamese_stopwords.txt   # Stopwords tiếng Việt cho CB Filter
```

## Thông Số Dữ Liệu Chính Thức

| Chỉ số | Giá trị |
|--------|---------|
| Tổng sản phẩm gốc | 49,688 |
| Sản phẩm non-food bị loại | 13,507 (27.2%) |
| **Sản phẩm food giữ lại** | **36,181** |
| Tổng records gốc | 33,819,108 |
| Records non-food bị loại | 1,899,791 (6.0%) |
| **Records food giữ lại** | **31,919,315** |
| Số đơn hàng sau lọc | 3,318,066 |

## Tài Liệu Liên Quan

| File | Mô tả |
|------|-------|
| `docs/data_survey.md` | Khảo sát chi tiết tập dữ liệu Instacart |
| `docs/models.md` | Mục đích dự án, phân biệt bài toán, chiến lược dữ liệu & phương pháp (mô tả thuật toán chi tiết) |

## Cách chạy Pipeline Model

```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Cài dependencies (làm 1 lần)
pip install -r requirements.txt

# Lưu ý: xóa các model cũ models/* trước khi chạy lại từ đầu

# Chạy từng bước (CD vào thư mục gốc của project)
python scripts/model/01_load_data.py
python scripts/model/02_cb_filter.py
python scripts/model/03_item_cf.py
python scripts/model/04_item_cf_neural.py
python scripts/model/05_kg_metapath.py
python scripts/model/06_ensemble.py
python scripts/model/07_eval_llm.py

# Hoặc chạy từ scripts/model/ (trong trường hợp đó sys.path tự xử lý)
cd scripts/model
python 01_load_data.py
python 02_cb_filter.py
...
```

## Lưu Ý

- Dữ liệu gốc (CSV) nằm trong `data/` và **không được commit** lên git
- `models/` và `results/` cũng được loại trừ qua `.gitignore`
- Chi tiết về dữ liệu xem tại [data_survey.md](data_survey.md)
- Chi tiết về mục đích dự án & model xem tại [models.md](models.md)
- **Các script không thuộc pipeline model (08, 09, 10, 12, 13, 14, 15...) không được đề cập trong docs chính** — chỉ tập trung vào pipeline 1→7