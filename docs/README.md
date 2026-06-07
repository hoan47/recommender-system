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
│   │   ├── build_confidence.py   # Confidence (Item-based CF) — Unified Scoring
│   │   ├── build_knowledge_graph.py  # Knowledge Graph (node2vec)
│   │   └── build_hybrid.py       # Hybrid: Confidence + KG, filter CB
│   └── evaluation/
│       └── evaluate.py           # Đánh giá models (CB, Confidence, KG, Hybrid)
├── models/                       # Output models (đã .gitignore)
├── results/                      # Kết quả evaluation (đã .gitignore)
├── english_stopwords.txt     # Danh sách stopword tiếng Anh (dùng cho CB)
├── requirements.txt
└── .gitignore
```

## Các model

| Model | File | Mục đích |
|-------|------|----------|
| CB | `build_cb.py` | TF-IDF dict-based, tìm sản phẩm tương tự (substitute), có thể dùng standalone hoặc làm bộ lọc |
| **Confidence** | **`build_confidence.py`** | **Unified scoring = ochiai × confidence × log1p, tìm sản phẩm mua kèm (complementary). BẤT ĐỐI XỨNG** |
| KG | `build_knowledge_graph.py` | Graph + node2vec, tìm sản phẩm liên quan qua đồ thị |
| Hybrid | `build_hybrid.py` | α*Confidence + β*KG, filter bằng CB |
| Tuning | `tune_hyperparams.py` | Grid search tự động 4 phase (CB → Confidence → KG → Hybrid) |

### Confidence — Unified Scoring (thay thế SPMI cũ)

**Công thức:**
```
ochiai(A,B) = cnt / sqrt(freq[A] * freq[B])   # Cosine Similarity
log_ab      = log1p(cnt)                        # Popularity Bonus
conf(A→B)   = cnt / freq[A]                     # Conditional Probability
score(A→B)  = ochiai * conf(A→B) * log_ab       # Unified Score
```

Đọc: "Nếu mua A, xác suất cũng mua B là bao nhiêu %, điều chỉnh theo độ tương quan (ochiai) và độ phổ biến (log)."

**Đặc điểm:**
- **BẤT ĐỐI XỨNG**: score(A→B) ≠ score(B→A) — phản ánh đúng hành vi mua kèm thực tế
- Kết hợp cả tần suất (popularity), xác suất có điều kiện (confidence), và mức độ tương quan (cosine similarity)
- Chỉ recommend từ sản phẩm có freq(A) ≥ FREQ_MIN (mặc định 30) — loại nhiễu từ sản phẩm quá hiếm

## Cấu hình Hybrid

Các tham số Hybrid được định nghĩa trong `src/config.py`:

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `HYBRID_ALPHA` | 0.2 | Trọng số Confidence (thay SPMI cũ) |
| `HYBRID_BETA` | 0.8 | Trọng số KG |
| `HYBRID_CB_THRESH` | 0.85 | Ngưỡng CB filter |

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
python src/features/build_confidence.py   # Unified Scoring (thay thế SPMI)
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
> Phase 1 (Confidence) nhanh hơn SPMI cũ vì không cần loop Numba JIT lần 2 cho scoring.