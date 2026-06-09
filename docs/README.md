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
│   ├── implementation_plan.md # Kế hoạch triển khai chi tiết
│   └── progress.md          # Bảng theo dõi tiến độ
├── models/                  # Model output — không commit
├── results/                 # Kết quả — không commit
├── scripts/                 # Script chạy từng bước riêng lẻ
│   ├── 01_load_data.py      # Load & cache dữ liệu
│   ├── 02_cb_filter.py      # Train CB Filter
│   ├── 03_ochiai.py         # Train Ochiai + Confidence
│   ├── 04_item2vec.py       # Train Item2Vec
│   ├── 05_node2vec.py       # Train Node2Vec
│   ├── 06_assoc_rules.py    # Train Association Rules
│   └── 07_ensemble.py       # Ensemble + test
├── src/                     # Source code
│   ├── config.py            # Cấu hình tập trung
│   ├── features/            # Xử lý dữ liệu & vector hóa
│   └── models/              # Các model recommendation
└── .gitignore               # Loại trừ data/, models/, results/
```

## Tài Liệu Liên Quan

| File | Mô tả |
|------|-------|
| `docs/data_survey.md` | Khảo sát chi tiết tập dữ liệu Instacart |
| `docs/models.md` | Mục đích dự án, phân biệt bài toán, chiến lược dữ liệu & phương pháp |
| `docs/implementation_plan.md` | Kế hoạch triển khai chi tiết — thứ tự code, module, pipeline |
| `docs/progress.md` | Bảng theo dõi tiến độ — hiện trạng file, model đã train |

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
python scripts/05_node2vec.py
python scripts/06_assoc_rules.py
python scripts/07_ensemble.py

# Hoặc chạy toàn bộ pipeline 1 lệnh
python run.py
```

## Lưu Ý

- Dữ liệu gốc (CSV) nằm trong `data/` và **không được commit** lên git
- `models/` và `results/` cũng được loại trừ qua `.gitignore`
- Chi tiết về dữ liệu xem tại [data_survey.md](data_survey.md)
- Chi tiết về mục đích dự án & model xem tại [models.md](models.md)

---