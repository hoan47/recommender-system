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
│   ├── tune_hyperparams.py       # Grid search tự động 4 phase
│   ├── features/
│   │   ├── build_cb.py           # Content-Based (standalone + substitute filter)
│   │   ├── build_spmi.py         # SPMI (collaborative filtering)
│   │   ├── build_knowledge_graph.py  # Knowledge Graph (node2vec)
│   │   └── build_hybrid.py       # Hybrid: SPMI + KG, filter CB
│   └── evaluation/
│       └── evaluate.py           # Đánh giá models (CB, SPMI, KG, Hybrid)
├── models/                       # Output models (đã .gitignore)
├── results/                      # Kết quả evaluation (đã .gitignore)
├── requirements.txt
└── .gitignore
```

## Các model

| Model | File | Mục đích |
|-------|------|----------|
| CB | `build_cb.py` | TF-IDF dict-based, tìm sản phẩm tương tự (substitute), có thể dùng standalone hoặc làm bộ lọc |
| SPMI | `build_spmi.py` | Co-occurrence → SPMI, tìm sản phẩm mua kèm (complementary) |
| KG | `build_knowledge_graph.py` | Graph + node2vec, tìm sản phẩm liên quan qua đồ thị |
| Hybrid | `build_hybrid.py` | α*SPMI + β*KG, filter bằng CB |
| Tuning | `tune_hyperparams.py` | Grid search tự động 4 phase (CB → SPMI → KG → Hybrid) |

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
| CB | — | — | — | — | — | — | — | — | — |
| SPMI | 1.38% | 2.43% | 3.89% | 0.34% | 0.36% | 0.41% | 0.18% | 0.14% | 0.12% |
| KG | 10.90% | 16.87% | 25.23% | 2.72% | 2.51% | 2.60% | 1.43% | 0.96% | 0.78% |
| Hybrid | **11.03%** | **16.97%** | **25.20%** | **2.80%** | **2.58%** | **2.67%** | **1.49%** | **1.01%** | **0.83%** |

> **Ghi chú:** Hybrid dùng α=0.2 (SPMI) + β=0.8 (KG), CB filter tạm tắt (threshold=1.0).
> CB chưa có kết quả — sẽ được cập nhật sau khi chạy tune_hyperparams.py.
> Metrics có thể được cải thiện sau khi tune.

## Chạy

### Build & đánh giá thủ công

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
```

### Tune hyperparameters tự động

```bash
# Grid search 4 phase: CB → SPMI → KG → Hybrid
python src/tune_hyperparams.py
```

Kết quả tune lưu tại `results/tune_results/`:
- `phase0_cb/` — snapshots từng tổ hợp CB params + best_params.json
- `phase1_spmi/` — snapshots từng tổ hợp SPMI params + best_params.json
- `phase2_kg/` — snapshots từng tổ hợp KG params + best_params.json
- `phase3_hybrid/` — snapshots từng tổ hợp Hybrid params + best_params.json
- `final_best_params.json` — tổng hợp best params toàn cục

### Grid search details

| Phase | Model | Tham số tune | Grid size |
|-------|-------|-------------|:---------:|
| 0 | CB | CB_MIN_DF, CB_MAX_DF, CB_MAX_FEATURES | 27 |
| 1 | SPMI | SPMI_K, SPMI_TOP_K | 12 |
| 2 | KG | KG_DIM, KG_WALK_LENGTH, KG_NUM_WALKS, KG_EPOCHS | 24 |
| 3 | Hybrid | HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH | ~40 |

> **Lưu ý:** Phase 2 (KG) là chậm nhất do phải train node2vec. Tổng thời gian có thể >24h nếu chạy full grid.
