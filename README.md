# Product Recommendation System

A product recommendation system comparing four approaches: Content-Based, Collaborative Filtering (SPMI), Knowledge Graph (RWR), and Hybrid. Built for basket completion — recommending complementary products rather than substitutes.

---

## Project Structure

```
recommendation-system/
│
├── models/                     # Recommendation models (split from all_model.py)
│   ├── __init__.py             # Public API: build_all, *_rec functions
│   ├── builder.py              # Orchestrates build sequence for all models
│   ├── content_based.py        # TF-IDF vectors + cosine similarity
│   ├── collaborative.py        # Co-occurrence + SPMI with cross-dept bonus
│   ├── knowledge_graph.py      # P→P graph + Random Walk with Restart (RWR)
│   ├── hybrid.py               # CF + KG scores, CB as substitute filter
│   └── utils.py                # Shared: diversity_filter
│
├── config.py                   # All paths and hyperparameters
├── data_loader.py              # Load CSVs, user-based train/test split
├── dept_direction.py           # Department association rules (Conf + Lift)
├── evaluate.py                 # Metrics: H@K, NDCG@K, P@K, R@K, F1@K
│
├── app.py                      # Streamlit web UI
├── run.py                      # CLI — train + evaluate all models
└── README.md
```

---

## Data

Place the following files in the directory defined by `DATA_DIR` in `config.py`:

| File | Description |
|---|---|
| `order_products_train.csv` | Prior order history (used as training data) |
| `order_products_test.csv` | Held-out test orders |
| `products.csv` | Product metadata (must include `product_name_vi`) |
| `departments.csv` | Department ID → name mapping |

---

## Setup

```bash
pip install streamlit pandas numpy tqdm psutil
```

---

## Running

### Web UI
```bash
streamlit run app.py
```

### CLI (train + full evaluation)
```bash
python run.py
```

---

## Models

### 1. Content-Based (`models/content_based.py`)
Builds a custom TF-IDF vector per product from Vietnamese product names (unigrams + bigrams). Recommends by cosine similarity. Used as a **substitute filter** in Hybrid — products that are too similar (cosine ≥ `CB_FILTER_HIGH`) are excluded from complementary recommendations.

### 2. Collaborative — SPMI (`models/collaborative.py`)
Computes Shifted Positive Mutual Information (SPMI) on product co-occurrence across baskets. Cross-department pairs receive a `CROSS_DEPT_BONUS` multiplier (default `1.5×`) to bias toward complementary products over substitutes.

### 3. Knowledge Graph — RWR (`models/knowledge_graph.py`)
Constructs a weighted P→P graph from SPMI edges (no aisle/department meta-nodes, by design). Scores candidates via Random Walk with Restart (RWR). Multi-hop traversal surfaces non-obvious complementary pairings.

### 4. Hybrid (`models/hybrid.py`)
Combines normalized CF (SPMI) and KG (RWR) scores with weight `HYB_ALPHA` (default `0.85`). Uses CB similarity to filter out substitutes. Applies a department multiplier to further boost cross-department candidates.

---

## Evaluation

Metrics computed at K = 10, 50, 100:

| Metric | Description |
|---|---|
| `H@K` | Hit Rate — fraction of test orders with ≥ 1 correct recommendation |
| `NDCG@K` | Normalized Discounted Cumulative Gain — rewards correct items ranked higher |
| `P@K` | Precision |
| `R@K` | Recall |
| `F1@K` | Harmonic mean of P and R |

**Benchmark (50k products, 99.9% sparsity):**

| H@10 | Assessment |
|---|---|
| < 0.10 | Needs improvement |
| ~ 0.15 | Acceptable |
| ~ 0.25 | Good |
| > 0.35 | Production-grade |

Test split: user-based — 80% earliest orders per user → train, 20% latest → test.

---

## Key Hyperparameters (`config.py`)

| Parameter | Default | Effect |
|---|---|---|
| `MIN_FREQ` | `50` | Minimum product purchase count to be included |
| `MAX_CASES` | `2000` | Number of test cases for evaluation |
| `KG_RWR_WALKS` | `1000` | RWR stability — higher = slower but more stable |
| `HYB_ALPHA` | `0.85` | CF weight in Hybrid (1 - alpha goes to KG) |
| `CB_FILTER_HIGH` | `0.80` | Cosine threshold above which a product is treated as a substitute |

---

## Extending the Project

To add a new model:

1. Create `models/my_model.py` with a `build(...)` and `recommend(product_id, k)` function.
2. Call `build(...)` inside `models/builder.py`.
3. Export `recommend` in `models/__init__.py`.
4. Add it to the `models_dict` in `run.py` and `app.py`.
