# 🧠 Recommender System — Instacart Market Basket Analysis

Hệ thống gợi ý sản phẩm dùng dataset Instacart Market Basket Analysis.

## Cấu trúc thư mục

```
recommender-system/
├── data/                         # Dữ liệu gốc (KHÔNG commit git)
├── docs/
│   ├── README.md                 # File này
│   ├── data_evaluation.md        # Đánh giá dữ liệu
│   └── new_plan.md               # Kế hoạch code
├── src/
│   ├── config.py                 # Cấu hình tập trung
│   ├── data_loader.py            # Load dữ liệu
│   ├── recommend.py              # Gợi ý sản phẩm
│   ├── features/
│   │   ├── build_cb.py           # Content-Based (substitute filter)
│   │   ├── build_spmi.py         # SPMI (collaborative filtering)
│   │   ├── build_knowledge_graph.py  # Knowledge Graph (node2vec)
│   │   └── build_hybrid.py       # Hybrid: SPMI + KG, filter CB
│   └── evaluation/
│       └── evaluate.py           # Đánh giá models
├── models/                       # Output models (đã .gitignore)
├── results/                      # Kết quả evaluation (đã .gitignore)
├── requirements.txt
└── .gitignore
```

## Các model

| Model | File | Mục đích |
|-------|------|----------|
| CB | `build_cb.py` (~35 dòng) | TF-IDF dict-based, tìm sản phẩm giống (substitute), dùng làm bộ lọc |
| SPMI | `build_spmi.py` (~55 dòng) | Co-occurrence → SPMI, tìm sản phẩm mua kèm (complementary) |
| KG | `build_knowledge_graph.py` (~100 dòng) | Graph + node2vec, tìm sản phẩm liên quan qua đồ thị |
| Hybrid | `build_hybrid.py` (~40 dòng) | α*SPMI + β*KG, filter bằng CB |

## Cấu hình Hybrid

Các tham số Hybrid được định nghĩa trong `src/config.py`:

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `HYBRID_ALPHA` | 0.2 | Trọng số SPMI (thấp vì SPMI recall chỉ ~1-4%) |
| `HYBRID_BETA` | 0.8 | Trọng số KG (cao vì KG recall ~11-25%) |
| `HYBRID_CB_THRESH` | 1.0 | Tạm tắt CB filter (sẽ tinh chỉnh sau) |

## Lưu ý Evaluation

Dataset Instacart public **KHÔNG cung cấp ground truth** cho test set (75K orders).
Do đó evaluation chạy trên **train set** (131K orders) với giao thức leave-one-out.

### Kết quả metrics hiện tại

| Model | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 | map@5 | map@10 | map@20 |
|-------|----------|-----------|-----------|--------|---------|---------|-------|--------|--------|
| SPMI | 1.38% | 2.43% | 3.89% | 0.34% | 0.36% | 0.41% | 0.18% | 0.14% | 0.12% |
| KG | 10.90% | 16.87% | 25.23% | 2.72% | 2.51% | 2.60% | 1.43% | 0.96% | 0.78% |
| Hybrid | **11.03%** | **16.97%** | **25.20%** | **2.80%** | **2.58%** | **2.67%** | **1.49%** | **1.01%** | **0.83%** |

> **Ghi chú:** Hybrid dùng α=0.2 (SPMI) + β=0.8 (KG), CB filter tạm tắt (threshold=1.0).

## Chạy

```bash
# 1. Build từng model theo thứ tự
python src/features/build_cb.py
python src/features/build_spmi.py
python src/features/build_knowledge_graph.py
python src/features/build_hybrid.py

# 2. Đánh giá
python src/evaluation/evaluate.py

# 3. Recommend thử
python src/recommend.py