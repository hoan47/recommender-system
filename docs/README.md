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
│   │   ├── build_spmi.py         # SPMI (DEPRECATED — giữ lại cho compat)
│   │   ├── build_confidence.py   # Confidence (Item-based CF) — THAY THẾ SPMI
│   │   ├── build_knowledge_graph.py  # Knowledge Graph (node2vec)
│   │   └── build_hybrid.py       # Hybrid: Confidence + KG, filter CB
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
| **Confidence** | **`build_confidence.py`** | **Co-occurrence → Confidence (Item-based CF), tìm sản phẩm mua kèm (complementary). THAY THẾ SPMI** |
| SPMI | `build_spmi.py` | Giữ lại cho backward compatibility |
| KG | `build_knowledge_graph.py` | Graph + node2vec, tìm sản phẩm liên quan qua đồ thị |
| Hybrid | `build_hybrid.py` | α*Confidence + β*KG, filter bằng CB |
| Tuning | `tune_hyperparams.py` | Grid search tự động 4 phase (CB → Confidence → KG → Hybrid) |

### Confidence — thuật toán mới thay SPMI

**Công thức:**
```
Confidence(A → B) = cooc(A,B) / freq(A)
```
Đọc: "Nếu mua A, xác suất cũng mua B là bao nhiêu %"

**Tại sao thay SPMI?**
- SPMI dùng `max(PMI - log(k), 0)` với threshold toàn cục → loại bỏ cặp phổ biến có PMI thấp (Burger+Khoai tây), giữ cặp hiếm ảo (Phô mai+Truffle chỉ 2 đơn chung)
- Confidence dùng % thực tế, không threshold, không log, không shift
- Chỉ recommend từ sản phẩm có freq(A) ≥ FREQ_MIN (mặc định 30) — loại nhiễu từ sản phẩm quá hiếm
- Ma trận KHÔNG đối xứng: Confidence(A→B) ≠ Confidence(B→A)

Ví dụ với Confidence:
| A | B | cooc | freq(A) | Confidence | Nhận xét |
|---|----|:---:|:-------:|:----------:|----------|
| Burger | Khoai tây | 8,000 | 20,000 | **40%** | Mua kèm mạnh |
| Cơm | Gà rán | 15,000 | 60,000 | **25%** | Mua kèm trung bình |
| Gà rán | Cơm | 15,000 | 20,000 | **75%** | Rất mạnh (ngược chiều) |
| Tương cà | Khoai tây chiên | 5,000 | 8,000 | **62.5%** | Rất mạnh |
| Phô mai ý | Sốt truffle | 2 | 5 | 40% → **KHÔNG recommend** | freq=5 < 30 |
| Dầu ăn | Nồi cơm điện | 40 | 35,000 | **0.11%** | Rất thấp → không recommend |

## Cấu hình Hybrid

Các tham số Hybrid được định nghĩa trong `src/config.py`:

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `HYBRID_ALPHA` | 0.2 | Trọng số Confidence (thay SPMI) |
| `HYBRID_BETA` | 0.8 | Trọng số KG |
| `HYBRID_CB_THRESH` | 1.0 | Tạm tắt CB filter (sẽ tinh chỉnh sau) |

### Tham số Confidence

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `CONF_FREQ_MIN` | 30 | Chỉ recommend từ sản phẩm có ≥30 đơn |
| `CONF_TOP_K` | 100 | Giữ tối đa 100 gợi ý mỗi sản phẩm |

## Lưu ý Evaluation

Dataset Instacart public **KHÔNG cung cấp ground truth** cho test set (75K orders).
Do đó evaluation chạy trên **train set** (131K orders) với giao thức leave-one-out.

> **Lưu ý:** Metrics cũ của SPMI (1-4% recall) không còn áp dụng. Cần chạy lại evaluation với Confidence để có số liệu mới.

## Chạy

### Build & đánh giá thủ công — Thứ tự mới

```bash
# 1. Build từng model theo thứ tự
python src/features/build_cb.py
python src/features/build_confidence.py    # <-- SPMI --> Confidence
python src/features/build_knowledge_graph.py
python src/features/build_hybrid.py

# 2. Đánh giá
python src/evaluation/evaluate.py

# 3. Recommend thử
python src/recommend.py
```

### Tune hyperparameters tự động

```bash
# Grid search 4 phase: CB → Confidence → KG → Hybrid
python src/tune_hyperparams.py
```

Kết quả tune lưu tại `results/tune_results/`:
- `phase0_cb/` — snapshots từng tổ hợp CB params + best_params.json
- `phase1_confidence/` — snapshots từng tổ hợp Confidence params + best_params.json
- `phase2_kg/` — snapshots từng tổ hợp KG params + best_params.json
- `phase3_hybrid/` — snapshots từng tổ hợp Hybrid params + best_params.json
- `final_best_params.json` — tổng hợp best params toàn cục

### Grid search details

| Phase | Model | Tham số tune | Grid size |
|-------|-------|-------------|:---------:|
| 0 | CB | CB_MIN_DF, CB_MAX_DF, CB_MAX_FEATURES | 27 |
| 1 | **Confidence** | CONF_FREQ_MIN, CONF_TOP_K | **12** |
| 2 | KG | KG_DIM, KG_WALK_LENGTH, KG_NUM_WALKS, KG_EPOCHS | 24 |
| 3 | Hybrid | HYBRID_ALPHA, HYBRID_BETA, HYBRID_CB_THRESH | ~40 |

> **Lưu ý:** Phase 2 (KG) là chậm nhất do phải train node2vec. Tổng thời gian có thể >24h nếu chạy full grid.
> Phase 1 (Confidence) nhanh hơn SPMI vì không cần loop Numba JIT lần 2.