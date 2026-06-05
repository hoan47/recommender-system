# 🧠 Recommender System — Instacart Market Basket Analysis

Hệ thống gợi ý sản phẩm sử dụng dataset **Instacart Market Basket Analysis**, hướng tiếp cận **Global (Item-Oriented)**.

## 🎯 Mục tiêu

Xây dựng ma trận quan hệ giữa các sản phẩm (item-item relationships) dựa trên:
- **Content-Based (CB)**: TF-IDF từ tên sản phẩm + ngành hàng → tìm sản phẩm **GIỐNG nhau** (substitute)
- **SPMI (Collaborative Filtering)**: Co-occurrence matrix → Shifted Positive PMI → tìm sản phẩm **MUA KÈM** (complementary)
- **Knowledge Graph (KG)**: Đồ thị product-department + co-purchase → node2vec embeddings → tìm sản phẩm **LIÊN QUAN**
- **Hybrid**: Kết hợp SPMI + KG, dùng CB làm bộ lọc loại substitute

## 📁 Cấu trúc thư mục

```
recommender-system/
├── .clinerules/                  # Quy tắc làm việc
├── data/                         # Dữ liệu gốc (KHÔNG commit git)
│   ├── aisles.csv                # (không dùng)
│   ├── departments.csv           # 21 ngành hàng
│   ├── orders.csv                # 3.4M đơn hàng
│   ├── order_products__prior.csv # 32.4M interactions (nguồn chính)
│   ├── order_products__train.csv # 1.38M labels
│   └── products.csv              # 49,688 sản phẩm
├── docs/                         # Tài liệu dự án
│   ├── README.md                 # 👈 File này
│   ├── data_evaluation.md        # Đánh giá dữ liệu
│   ├── models.md                 # Kiến trúc models
│   └── implementation_plan.md    # Kế hoạch implement
├── src/                          # Source code
│   ├── features/
│   │   ├── build_tfidf.py        # CB
│   │   ├── build_spmi.py         # SPMI
│   │   ├── build_knowledge_graph.py  # KG
│   │   └── build_hybrid.py       # Hybrid
│   ├── evaluation/
│   │   └── evaluate.py           # Đánh giá models
│   └── utils/
│       └── data_loader.py        # Load dữ liệu
├── models/                       # Output models (đã .gitignore)
├── results/                      # Kết quả evaluation (đã .gitignore)
├── requirements.txt              # Python dependencies
└── .gitignore
```

## ⚙️ Yêu cầu hệ thống

- Python 3.9+
- RAM tối thiểu: 8GB (khuyến nghị 16GB)
- Dung lượng ổ cứng: ~5GB cho dữ liệu + models

## 🚀 Cài đặt & Chạy

```bash
# 1. Clone repo
git clone https://github.com/hoan47/recommender-system.git
cd recommender-system

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Đảm bảo dữ liệu đã ở thư mục data/

# 5. Chạy từng bước theo thứ tự
python src/utils/data_loader.py
python src/features/build_tfidf.py          # CB
python src/features/build_spmi.py           # SPMI + tune + eval
python src/features/build_knowledge_graph.py # KG + tune + eval
python src/features/build_hybrid.py         # Hybrid + tune + eval
python src/evaluation/evaluate.py           # So sánh tất cả models
```

## 📊 Dữ liệu

| File | Records | Vai trò |
|------|---------|---------|
| `products.csv` | 49,688 | Danh sách sản phẩm + department |
| `departments.csv` | 21 | Ngành hàng |
| `orders.csv` | 3,421,083 | Thông tin đơn hàng (có cột eval_set) |
| `order_products__prior.csv` | 32,434,489 | Lịch sử mua hàng — **XÂY MODEL** |
| `order_products__train.csv` | 1,384,617 | **TUNE** hyperparameters |
| `order_products__test.csv` | (trong train) | **EVAL** cuối cùng |

> **File `aisles.csv` không được sử dụng.** Aisle là vị trí vật lý (kệ hàng), khác nhau giữa các cửa hàng trong cùng chuỗi, không phải phân loại toàn cục.

## 🧪 Metrics đánh giá

| Metric | Mô tả |
|--------|-------|
| **Recall@K** | Tỉ lệ sản phẩm đúng trong top-K gợi ý |
| **NDCG@K** | Độ chính xác có xét đến thứ hạng |
| **MAP@K** | Trung bình average precision trên tất cả queries |

## 📦 Output

Tất cả output được lưu trong thư mục `models/`:
- `models/tfidf_matrix.npz` — TF-IDF vectors (CB)
- `models/item_similarity_cb.npz` — Cosine similarity (CB)
- `models/cooc_matrix.npz` — Co-occurrence counts (SPMI)
- `models/spmi_matrix.npz` — SPMI values (SPMI)
- `models/spmi_best_k.json` — Threshold tối ưu (SPMI)
- `models/kg_embeddings.npy` — Product embeddings (KG)
- `models/kg_best_params.json` — Params tối ưu (KG)
- `models/hybrid_weights.json` — Trọng số α, β (Hybrid)
- `results/metrics.json` — Kết quả đánh giá

## 📝 Ghi chú

- Dữ liệu gốc không được commit lên git (đã có .gitignore)
- File CSV cần đọc với encoding `utf-8`
- Dùng `csv.DictReader` thay vì split tay (có dấu phẩy trong ngoặc kép)
- `order_products__prior.csv` 32.4M records → xử lý theo chunk
- Ma trận co-occurrence 50K×50K → bắt buộc dùng sparse matrix (scipy.sparse)

## 📚 Tham khảo

- [Instacart Market Basket Analysis - Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis)
- [docs/data_evaluation.md](data_evaluation.md)
- [docs/models.md](models.md)
- [docs/implementation_plan.md](implementation_plan.md)