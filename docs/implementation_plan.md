# Kế Hoạch Triển Khai (Implementation Plan)

> ⚠️ **Nguyên tắc:** Tuân theo đúng thứ tự trong [models.md — Mục 4.8](models.md#48-thứ-tự-triển-khai). Mỗi bước chạy độc lập, có thể kiểm tra trước khi sang bước tiếp theo.
>
> **Lưu ý:** Evaluation (Survey + LLM + Metrics) sẽ được thiết kế và triển khai ở giai đoạn sau, không nằm trong kế hoạch này.

---

## 1. Cấu trúc thư mục source code cuối cùng

```
src/
├── __init__.py
│
├── config.py                       # Hyperparameters, paths, constants
│
├── features/
│   ├── __init__.py
│   ├── loader.py                   # Đọc CSV, merge dữ liệu
│   └── vectorizer.py               # TF-IDF + one-hot cho CB Filter
│
└── models/
    ├── __init__.py
    ├── cb_filter.py                # Content-Based Diversity Filter (Bước 1)
    ├── ochiai.py                   # Ochiai + Confidence Score (Bước 2)
    ├── item2vec.py                 # Item2Vec Word2Vec Skip-gram (Bước 3)
    ├── node2vec.py                 # Node2Vec graph embedding (Bước 4)
    ├── assoc_rules.py              # Association Rules từ co-occurrence matrix (Bước 5)
    └── ensemble.py                 # Co-occurrence Ensemble + CB Filter (Bước 6-7)
```

---

## 2. Mô tả chi tiết từng module

### 2.0 `src/config.py` — Cấu hình tập trung

**Mục đích:** Mọi hyperparameter, đường dẫn, hằng số được đặt tại 1 file, tránh hardcode.

**Nội dung:**

```python
# Đường dẫn dữ liệu
DATA_DIR = "data/"
PROCESSED_DIR = "data/processed/"
MODEL_DIR = "models/"
RESULT_DIR = "results/"

# Files gốc
ORDERS_FILE = DATA_DIR + "orders.csv"
PRODUCTS_FILE = DATA_DIR + "products.csv"
ORDER_PRODUCTS_PRIOR = DATA_DIR + "order_products__prior.csv"
ORDER_PRODUCTS_TRAIN = DATA_DIR + "order_products__train.csv"
AISLES_FILE = DATA_DIR + "aisles.csv"
DEPARTMENTS_FILE = DATA_DIR + "departments.csv"

# Files processed (lưu sau khi preprocess)
COOCCURRENCE_FILE = PROCESSED_DIR + "cooccurrence.npz"
PRODUCT_VECTORS_FILE = PROCESSED_DIR + "product_vectors.npz"

# Hyperparameters — CB Filter
CB_THRESHOLD = 0.8          # cosine similarity ≥ threshold → substitute → loại
CB_N_GRAM_RANGE = (1, 2)    # TF-IDF: unigram + bigram
CB_MAX_FEATURES = 5000      # TF-IDF: max số features từ tên sản phẩm

# Hyperparameters — Ochiai
OCHIAI_MIN_SUPPORT = 30     # pair xuất hiện < 30 lần → bỏ qua
OCHIAI_TOP_K = 100          # số candidate giữ lại cho mỗi product (trước ensemble)

# Hyperparameters — Item2Vec
I2V_VECTOR_SIZE = 128
I2V_WINDOW = 10
I2V_MIN_COUNT = 10
I2V_NEGATIVE = 10
I2V_EPOCHS = 20
I2V_WORKERS = 4
I2V_TOP_K = 100

# Hyperparameters — Node2Vec
N2V_EMBEDDING_DIM = 128
N2V_WALK_LENGTH = 40
N2V_NUM_WALKS = 20
N2V_P = 1.0
N2V_Q = 1.0
N2V_WORKERS = 4
N2V_EDGE_THRESHOLD = 5          # edge giữa 2 product nếu co-occur ≥ threshold (count raw)
N2V_TOP_K = 100

# Hyperparameters — Association Rules (tự implement từ co-occurrence matrix)
ARM_MIN_SUPPORT = 0.0001        # support threshold (tỷ lệ, không phải số tuyệt đối)
ARM_MIN_CONFIDENCE = 0.1        # confidence threshold
ARM_MIN_LIFT = 1.5              # lift threshold
ARM_TOP_K = 100

# Hyperparameters — Ensemble
ENS_ALPHA = 0.5                 # trọng số Ochiai
ENS_BETA  = 0.25                # trọng số Item2Vec
ENS_GAMMA = 0.25                # trọng số Node2Vec
ENS_TOP_K = 100                 # top-K sau ensemble (trước CB filter)
ENS_FINAL_K = 10                # top-K cuối cùng output
```

---

### 2.1 `src/features/loader.py` — Đọc & merge dữ liệu

**Input:** Các file CSV gốc trong `data/`
**Output:** DataFrame products (đã merge aisle + department), danh sách order-product

**Function list:**

```python
def load_products() -> pd.DataFrame
    """
    Đọc products.csv, merge với aisles.csv và departments.csv.
    Return: DataFrame columns: [product_id, product_name, aisle_id, 
                                aisle, department_id, department]
    """

def load_order_products() -> pd.DataFrame
    """
    Đọc order_products__prior.csv + order_products__train.csv.
    Dùng chunksize=500K, concat thành 1 DataFrame.
    Return: DataFrame columns: [order_id, product_id, add_to_cart_order, reordered]
    """

def load_orders(eval_set: str = None) -> pd.DataFrame
    """
    Đọc orders.csv, lọc theo eval_set nếu có.
    eval_set=None → trả về tất cả.
    Return: DataFrame columns: [order_id, user_id, eval_set, order_number, ...]
    """

def get_product_name_map() -> dict
    """
    Return: {product_id: product_name} — dùng cho logging/display.
    """
```

---

### 2.2 `src/features/vectorizer.py` — TF-IDF + One-hot cho CB Filter

**Input:** DataFrame products (từ loader.load_products())
**Output:** Ma trận sparse (n_products × n_features)

**Function list:**

```python
def build_product_vectors(products_df: pd.DataFrame,
                          ngram_range=(1,2),
                          max_features=5000) -> tuple[sparse.csr_matrix, TfidfVectorizer]
    """
    Vector hóa sản phẩm:
      - TF-IDF trên product_name (unigram + bigram)
      - One-hot aisle_id
      - One-hot department_id
    Ghép dọc 3 vector lại bằng hstack.
    
    Return: (product_vectors: sparse.csr_matrix shape (n_products, D),
             vectorizer: TfidfVectorizer)
    """

def cb_similarity(product_vectors: sparse.csr_matrix,
                  product_a_id: int,
                  candidate_ids: list[int]) -> np.ndarray
    """
    Tính cosine similarity giữa product_a và từng candidate.
    Chỉ tính on-demand, không pre-compute full matrix.
    
    Args:
        product_vectors: ma trận (n_products, D)
        product_a_id: index của product A
        candidate_ids: list index của các candidate
    
    Return: numpy array shape (len(candidate_ids),) — similarity scores [0,1]
    """
```

---

### 2.3 `src/models/cb_filter.py` — Content-Based Diversity Filter

**Input:** 
  - product_vectors từ vectorizer
  - (product_A_id, list_candidate_ids, list_candidate_scores) từ ensemble
**Output:** list candidate đã loại bỏ substitute, giữ nguyên thứ tự score

```python
class CBFilter:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.product_vectors = None
        self.product_id_to_idx = {}  # mapping product_id → row index in matrix
    
    def fit(self, products_df: pd.DataFrame,
            ngram_range=(1,2), max_features=5000) -> None
        """
        Pre-compute product vectors 1 lần.
        Lưu product_vectors và mapping product_id → idx.
        """
    
    def filter(self, product_a_id: int,
               candidates: list[tuple[int, float]]) -> list[tuple[int, float]]
        """
        Lọc substitute khỏi danh sách candidate.
        
        Args:
            product_a_id: product_id đầu vào
            candidates: list (product_id, score) đã sort giảm dần theo score
        
        Return:
            list (product_id, score) đã loại bỏ các product có 
            CB_similarity >= threshold
        """
    
    def filter_df(self, product_a_id: int,
                  candidate_df: pd.DataFrame,
                  score_col: str = "score") -> pd.DataFrame
        """Version trả về DataFrame thay vì list."""
```

---

### 2.4 `src/models/ochiai.py` — Ochiai + Confidence Score

**Input:** order_products DataFrame
**Output:** Ochiai model có method `recommend(product_id, top_k)` và `save/load`

**Thiết kế:**
- Xây CSR co-occurrence matrix (n_products × n_products)
- Tính count vector cho mỗi product
- Khi recommend: tính score cho tất cả candidate, sort, lấy top-K
- Lưu ma trận sparse ra file `.npz` để dùng lại

```python
class OchiaiModel:
    def __init__(self, min_support: int = 30):
        self.min_support = min_support
        self.cooc_matrix = None         # CSR (n_products, n_products) — co-occurrence counts
        self.product_counts = None      # array (n_products,) — count(A)
        self.n_products = 0
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
    
    def fit(self, order_products: pd.DataFrame,
            products_df: pd.DataFrame) -> None
        """
        Xây co-occurrence matrix từ order_products.
        
        Steps:
          1. Map product_id → idx (0..n_products-1)
          2. Duyệt order_products theo order_id:
             - Với mỗi order có n items:
               - Duyệt all pairs (i,j) với i < j
               - Tăng cooc_matrix[i,j] += 1 và cooc_matrix[j,i] += 1
          3. Áp dụng min_support: zero-out các cell < min_support
          4. Tính product_counts riêng bằng Counter.
        
        ⚠️ Xử lý 33M records:
          - Dùng chunksize=500K khi đọc order_products
          - Dùng defaultdict(int) để đếm cặp trước, sau đó chuyển sang CSR
          - Hoặc dùng numba + array để tăng tốc đếm cặp
        """
    
    def _compute_scores(self, product_idx: int) -> np.ndarray
        """
        Tính score cho product_idx với tất cả các product khác.
        
        score(i→j) = ochiai(i,j) * conf(i→j) * log1p(cnt(i,j))
        
        Return: array (n_products,) — score từ product_idx đến mọi product
        """
    
    def recommend(self, product_id: int, top_k: int = 100) -> list[tuple[int, float]]
        """
        Trả về top-K product có score cao nhất cho product_id đầu vào.
        Return: list (product_id, score) sorted giảm dần.
        """
    
    def save(self, path: str) -> None
        """Lưu cooc_matrix (npz) + metadata (json)"""
    
    def load(self, path: str) -> None
        """Load từ file npz + json"""
```

---

### 2.5 `src/models/item2vec.py` — Item2Vec

**Input:** order_products DataFrame
**Output:** Gensim Word2Vec model + wrapper

```python
class Item2VecModel:
    def __init__(self, vector_size=128, window=10, min_count=10,
                 negative=10, epochs=20, workers=4):
        self.params = {...}
        self.model = None           # gensim.models.Word2Vec
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
    
    def fit(self, order_products: pd.DataFrame,
            products_df: pd.DataFrame) -> None
        """
        1. Group order_products theo order_id → list of product_ids per order
        2. Map product_id → str (gensim yêu cầu tagged document)
        3. Train Word2Vec (SG, negative sampling)
        
        ⚠️ Xử lý 33M records:
          - Duyệt chunk để build list of lists (không load hết vào memory)
          - Mỗi order là 1 list product_id strings
        """
    
    def recommend(self, product_id: int, top_k: int = 100) -> list[tuple[int, float]]
        """
        Dùng model.wv.most_similar() để lấy top-K product gần nhất.
        Return: list (product_id, cosine_similarity)
        """
    
    def save(self, path: str) -> None
        """Lưu gensim model + mapping"""
    
    def load(self, path: str) -> None
        """Load gensim model + mapping"""
```

---

### 2.6 `src/models/node2vec.py` — Node2Vec

**Input:** 
  - order_products DataFrame (để xây graph)
  - products_df (để mapping)
**Output:** Node embeddings + wrapper

**Thiết kế:**
- Xây graph dùng networkx
- Edge weight = co-occurrence count raw (≥ edge_threshold)
- Edge threshold: chỉ giữ edge >= 5 (co-occurrence count raw)
- Random walk + Word2Vec

```python
class Node2VecModel:
    def __init__(self, embedding_dim=128, walk_length=40, num_walks=20,
                 p=1.0, q=1.0, edge_threshold=5, workers=4):
        self.params = {...}
        self.graph = None
        self.model = None
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
    
    def fit(self, order_products: pd.DataFrame,
            products_df: pd.DataFrame) -> None
        """
        1. Build co-occurrence graph:
           - Node: mỗi sản phẩm
           - Edge: nếu co-occurrence count >= edge_threshold
           - Weight: co-occurrence count (raw)
        2. Random walk trên graph (tạo sentences)
        3. Train Word2Vec trên các sentences
        
        ⚠️ 49K nodes:
          - Edge count có thể rất lớn (hàng triệu)
          - Dùng networkx Graph, cần kiểm tra memory
          - Edge threshold = 5 giúp giảm số edge
        """
    
    def recommend(self, product_id: int, top_k: int = 100) -> list[tuple[int, float]]
        """
        Cosine similarity trên embedding space.
        Return: list (product_id, similarity)
        """
    
    def save(self, path: str) -> None
        """Lưu embeddings + graph + mapping"""
    
    def load(self, path: str) -> None
        """Load từ file"""
```

---

### 2.7 `src/models/assoc_rules.py` — Association Rules (tự implement từ co-occurrence matrix)

**Input:** Co-occurrence matrix + product_counts từ OchiaiModel
**Output:** DataFrame rules (antecedent, consequent, support, confidence, lift)

**Quyết định thiết kế:**
- **Không dùng mlxtend FP-Growth**: Vì 3.4M orders × 49K products không khả thi.
- **Tự implement từ co-occurrence matrix** đã có sẵn từ OchiaiModel:
  - `support(A,B)` = `cooc[A,B] / total_orders`
  - `confidence(A→B)` = `cooc[A,B] / count(A)`
  - `lift(A,B)` = `confidence(A→B) / support(B)`
  - Chạy được trên **toàn bộ dữ liệu** vì chỉ cần query cooc_matrix.

```python
class AssocRulesModel:
    def __init__(self, min_support=0.0001, min_confidence=0.1,
                 min_lift=1.5, top_k=100):
        self.params = {...}
        self.total_orders = None     # tổng số orders (lấy từ dữ liệu)
        self.cooc_matrix = None      # CSR matrix từ OchiaiModel (chia sẻ)
        self.product_counts = None   # array (n_products,) — count(A)
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self.rules_df = None         # DataFrame rules đã lọc
    
    def fit(self, ochiai_model: OchiaiModel, order_products: pd.DataFrame) -> None
        """
        Dùng cooc_matrix + product_counts từ OchiaiModel đã train.
        
        1. Tính total_orders = order_products['order_id'].nunique()
        2. Với mỗi product A (có count(A) > 0):
           - Lấy các product B có cooc[A,B] > 0
           - Tính support, confidence, lift cho từng cặp
           - Filter theo min_support, min_confidence, min_lift
        3. Lưu thành DataFrame rules
        
        ⚠️ Xử lý 49K products:
          - Không duyệt full matrix O(n²) — chỉ duyệt non-zero cells
          - CSR matrix.getrow() để lấy danh sách B có co-occurrence > 0
        """
    
    def recommend(self, product_id: int, top_k: int = 100) -> list[tuple[int, float]]
        """
        Tìm rules có antecedent = product_id.
        Sort theo lift descending.
        Return: list (product_id, lift)
        """
    
    def save(self, path: str) -> None
        """Lưu rules_df ra CSV"""
    
    def load(self, path: str) -> None
        """Load rules_df từ CSV"""
```

---

### 2.8 `src/models/ensemble.py` — Co-occurrence Ensemble + CB Filter

**Input:** Các model đã train (Ochiai, Item2Vec, Node2Vec) + CB Filter
**Output:** Top-K gợi ý cuối cùng

```python
class EnsembleModel:
    def __init__(self, alpha=0.5, beta=0.25, gamma=0.25,
                 top_k=100, final_k=10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k         # số candidate trước CB filter
        self.final_k = final_k     # số output cuối cùng
        self.ochiai = None
        self.item2vec = None
        self.node2vec = None
        self.cb_filter = None
    
    def fit(self, ochiai: OchiaiModel, 
            item2vec: Item2VecModel,
            node2vec: Node2VecModel,
            cb_filter: CBFilter) -> None
        """Gán các model đã train."""
    
    def _normalize(self, scores: list[float]) -> list[float]
        """Min-max normalize scores về [0, 1]."""
    
    def recommend(self, product_id: int, 
                  use_cb_filter: bool = True) -> list[tuple[int, float]]
        """
        1. Lấy top-K candidate từ mỗi model (O, I, N)
        2. Union các candidate lại
        3. Tính weighted score cho mỗi candidate
        4. Sort descending
        5. (Optional) CB Filter loại substitute
        6. Trả về final_k gợi ý
        
        Return: list (product_id, final_score)
        """
    
    def recommend_all(self, product_ids: list[int],
                      use_cb_filter: bool = True) -> dict[int, list[tuple[int, float]]]
        """Batch recommend cho nhiều product."""
```

---

## 3. Thứ tự chạy (Pipeline)

```
Bước 0: pip install -r requirements.txt (cần cập nhật thêm gensim, scikit-learn)

Bước 1: src/features/loader.py → load data
        src/features/vectorizer.py → build product vectors
        → Lưu: data/processed/product_vectors.npz

Bước 2: src/models/cb_filter.py → fit()
        → Cần product_vectors từ Bước 1

Bước 3: src/models/ochiai.py → fit()
        → Xây co-occurrence matrix (dùng numba/defaultdict)
        → Lưu: models/ochiai/ (cooc_matrix.npz + metadata.json)
        → Output này được dùng lại cho AssocRulesModel

Bước 4: src/models/item2vec.py → fit()
        → Lưu: models/item2vec/ (word2vec.model + mapping.json)

Bước 5: src/models/node2vec.py → fit()
        → Xây graph từ co-occurrence count, edge threshold=5
        → Lưu: models/node2vec/ (embeddings.npy + mapping.json)

Bước 6: src/models/assoc_rules.py → fit()
        → Dùng cooc_matrix từ OchiaiModel
        → Lưu: models/assoc_rules/ (rules.csv)

Bước 7: src/models/ensemble.py → fit()
        → Load 3 model + CB Filter → sẵn sàng recommend

Bước 8: Chạy thử recommend() với một số sản phẩm mẫu
        → So sánh ensemble with/without CB Filter
        → Ghi log kết quả để kiểm tra bằng mắt thường
```

---

## 4. File đầu ra dự kiến

| File | Mô tả | Kích thước ước tính |
|------|-------|---------------------|
| `data/processed/product_vectors.npz` | Ma trận product vectors (CSR) | ~50 MB |
| `models/ochiai/cooc_matrix.npz` | Co-occurrence sparse matrix | ~50-200 MB (cần kiểm tra) |
| `models/ochiai/metadata.json` | Mapping + params | ~1 MB |
| `models/ochiai/product_counts.csv` | Số lần xuất hiện của mỗi product | ~1 MB |
| `models/item2vec/word2vec.model` | Gensim model | ~100 MB |
| `models/item2vec/mapping.json` | product_id ↔ idx | ~1 MB |
| `models/node2vec/embeddings.npy` | Node embeddings (49K × 128) | ~25 MB |
| `models/assoc_rules/rules.csv` | Association rules (filtered) | ~100 MB |

---

## 5. Dependencies cần thêm vào requirements.txt

```text
# Hiện có:
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
networkx>=2.6.0
tqdm>=4.62.0
psutil>=5.8.0
numba>=0.56.0

# Cần thêm:
scikit-learn>=1.0.0          # TF-IDF, cosine_similarity
gensim>=4.2.0                # Word2Vec
```

> **Không cần mlxtend**: Association Rules được implement trực tiếp từ co-occurrence matrix (OchiaiModel), chạy được trên full dataset.

---

## 6. Những lưu ý kỹ thuật quan trọng

### 6.1 Xử lý memory cho co-occurrence matrix
- 49K × 49K = 2.4 tỷ cells
- Dense → ~20 GB
- CSR với density ước tính ~0.1% → ~2.4M non-zero → ~50 MB
- **Quan trọng:** Không dùng dense matrix, không dùng DataFrame pivot

### 6.2 Xử lý chunk cho 33M records
- Dùng `pd.read_csv(chunksize=500000)`
- Đếm cặp (A, B) bằng `defaultdict(int)` (hoặc `numba` nếu cần tối ưu)
- Cập nhật dần dần vào CSR? → **Không**, build từ defaultdict → CSR 1 lần

### 6.3 CB Filter không cần pre-compute full matrix
- Chỉ tính cosine similarity on-demand cho top-100 candidates
- Cost: 100 phép tính dot product sparse → rất nhanh

### 6.4 Association Rules — tự implement từ co-occurrence matrix
- **Cơ sở:** OchiaiModel đã xây co-occurrence matrix (CSR) + product_counts
- **Support(A,B):** `cooc[A,B] / total_orders`
- **Confidence(A→B):** `cooc[A,B] / count(A)`
- **Lift(A,B):** `confidence(A→B) / (count(B) / total_orders)`
- **Ưu điểm:** Không cần FP-Growth, không cần subsample, chạy được trên full dataset
- Chỉ duyệt non-zero cells của CSR matrix — hiệu quả với sparse matrix

### 6.5 Node2Vec — edge weight = co-occurrence count raw
- Edge threshold = 5 (co-occurrence count ≥ 5)
- Weight = count raw (không dùng Ochiai score)
- Đơn giản, trực quan, không phụ thuộc vào OchiaiModel

---

## 7. Bảng tiến độ (Progress Tracking)

| Bước | File | Trạng thái |
|------|------|-----------|
| 0 | `requirements.txt` (cập nhật dependencies) | ⬜ |
| 1 | `src/config.py` | ⬜ |
| 2 | `src/features/loader.py` | ⬜ |
| 3 | `src/features/vectorizer.py` | ⬜ |
| 4 | `src/models/cb_filter.py` | ⬜ |
| 5 | `src/models/ochiai.py` | ⬜ |
| 6 | `src/models/item2vec.py` | ⬜ |
| 7 | `src/models/node2vec.py` | ⬜ |
| 8 | `src/models/assoc_rules.py` | ⬜ |
| 9 | `src/models/ensemble.py` | ⬜ |

---

## 8. Kế hoạch commit sau mỗi bước

| Commit type | Khi nào |
|-------------|---------|
| `[backup] config + features: loader và vectorizer` | Sau bước 1-3 |
| `[feat] cb filter + ochiai model` | Sau bước 4-5 |
| `[feat] item2vec model: gensim word2vec` | Sau bước 6 |
| `[feat] node2vec model: graph embedding` | Sau bước 7 |
| `[feat] assoc_rules baseline: from cooc matrix` | Sau bước 8 |
| `[feat] ensemble + cb filter integration` | Sau bước 9 |