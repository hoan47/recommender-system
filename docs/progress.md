# 📊 Bảng hiện trạng file

Cập nhật: 6/8/2026

## Cấu trúc thư mục

```
recommender-system/
├── data/                           # Dữ liệu gốc (đã .gitignore)
├── docs/
│   ├── README.md                   # ✅ Tài liệu chính
│   ├── data_evaluation.md          # ✅ Đánh giá dữ liệu
│   └── progress.md                 # ✅ File này
├── models/                         # Output models (đã .gitignore)
├── results/                        # Kết quả evaluation (đã .gitignore)
├── src/
│   ├── __init__.py                 # ✅ Package init
│   ├── config.py                   # ✅ Cấu hình tập trung
│   ├── data_loader.py              # ✅ Load dữ liệu + temporal split
│   ├── recommend.py                # ✅ Gợi ý sản phẩm
│   ├── tune_hyperparams.py         # ✅ Grid search 4 phase
│   ├── evaluation/
│   │   └── evaluate.py             # ✅ Đánh giá models (refactored)
│   └── features/
│       ├── build_cb.py             # ✅ Content-Based
│       ├── build_association_rules.py  # ✅ Association Rules
│       ├── build_knowledge_graph.py    # ✅ Knowledge Graph
│       └── build_hybrid.py         # ✅ Hybrid
├── english_stopwords.txt           # ✅ Stopword cho CB
├── requirements.txt                # ✅ Dependencies
└── .gitignore                      # ✅ Git ignore
```

## Trạng thái các file

| File | Trạng thái | Ghi chú |
|------|-----------|---------|
| `src/config.py` | ✅ Hoàn thành | Thêm `PATH_OUTPUT_CSV` cho evaluate.py mới |
| `src/data_loader.py` | ✅ Hoàn thành | Thêm `load_temporal_test_cases()`, `_build_user_orders()`, `_split_user_orders()` |
| `src/evaluation/evaluate.py` | ✅ Refactored | Thiết kế mới theo file tham khảo: `calc_metrics()`, `evaluate_one()`, `run_comparison()` |
| `docs/README.md` | ✅ Cập nhật | Mục Evaluation mới với temporal split + benchmark guide |
| `docs/progress.md` | ✅ Tạo mới | File này |

## Các thay đổi gần đây

### 6/8/2026 — Refactor evaluation

- **Vấn đề:** File evaluate.py cũ eval trên **train set** (131K orders) vì cho rằng test set không có ground truth — điều này sai.
- **Giải pháp:** Implement temporal user-based split (80% order đầu → train, 20% cuối → test) đảm bảo không leakage.
- **Thiết kế mới** dựa trên evaluate.py tham khảo:
  - Generic interface: `rec_func(seed, top_k)` → dễ test nhiều model
  - Tách biệt H@K (Hit Rate) và R@K (Recall)
  - Thêm benchmark guide
  - Output DataFrame có rank, best model, improvement %
  - Lưu CSV + JSON
- **File mới/thay đổi:**
  - `src/config.py`: thêm `PATH_OUTPUT_CSV`
  - `src/data_loader.py`: thêm temporal split functions
  - `src/evaluation/evaluate.py`: viết lại hoàn toàn