"""
Cấu hình tập trung cho toàn bộ dự án Recommender System.

Chứa tất cả hằng số: đường dẫn, tham số model, tên file output, v.v.
Các file khác import module này thay vì định nghĩa lại hằng số cục bộ.
"""

import json
from pathlib import Path

# ============================================================================
# ĐƯỜNG DẪN
# ============================================================================

# Thư mục gốc dự án (tính từ src/config.py → lên 2 level = recommender-system/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Tự động tạo thư mục khi import module
for _dir in [MODELS_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DỮ LIỆU
# ============================================================================

DATA_ENCODING = "utf-8"
PRIOR_CHUNK_SIZE = 500000

# ============================================================================
# CB (Content-Based) — TF-IDF + Cosine Similarity
# ============================================================================

CB_MAX_FEATURES = 10000
CB_TOP_K = 100
CB_CHUNK_SIZE = 1000

ENGLISH_STOP_WORDS = frozenset({
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "bill",
    "both",
    "bottom",
    "but",
    "by",
    "call",
    "can",
    "cannot",
    "cant",
    "co",
    "con",
    "could",
    "couldnt",
    "cry",
    "de",
    "describe",
    "detail",
    "do",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eg",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fifty",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "interest",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "ltd",
    "made",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "rather",
    "re",
    "same",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "system",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thick",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "un",
    "under",
    "until",
    "up",
    "upon",
    "us",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves"
})

# ============================================================================
# SPMI (Collaborative Filtering — Shifted Positive PMI)
# ============================================================================

SPMI_K_VALUES = (1, 2, 3, 5, 10)
SPMI_TOTAL_PRIOR_ORDERS = 3214874
SPMI_EVAL_KS = (5, 10, 20)

# ============================================================================
# KG (Knowledge Graph) — Node2Vec
# ============================================================================

KG_WALK_LENGTHS = (10, 20, 30)
KG_DIMENSIONS_LIST = (64, 128)
KG_NUM_WALKS_LIST = (100, 200)
KG_WINDOW = 10
KG_P = 1.0
KG_Q = 1.0
KG_EPOCHS = 1
KG_NEGATIVE = 5
KG_LR = 0.025
KG_TOP_K = 100
KG_CHUNK_SIZE = 1000
KG_EVAL_KS = (5, 10, 20)

# ============================================================================
# HYBRID — Kết hợp SPMI + KG, lọc bởi CB
# ============================================================================

HYBRID_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
HYBRID_BETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
HYBRID_CB_THRESHOLDS = [0.7, 0.8, 0.9]
HYBRID_EVAL_KS = (5, 10, 20)

# ============================================================================
# EVALUATION
# ============================================================================

EVAL_KS = (5, 10, 20)

# ============================================================================
# TÊN FILE OUTPUT
# ============================================================================

# CB
TFIDF_MATRIX_FILE = "tfidf_matrix.npz"
CB_SIMILARITY_FILE = "item_similarity_cb.npz"
TFIDF_VOCAB_FILE = "tfidf_vocab.json"

# SPMI
COOC_MATRIX_FILE = "cooc_matrix.npz"
SPMI_MATRIX_FILE = "spmi_matrix.npz"
SPMI_BEST_K_FILE = "spmi_best_k.json"

# KG
KG_EMBEDDINGS_FILE = "kg_embeddings.npy"
KG_BEST_PARAMS_FILE = "kg_best_params.json"
KG_SIMILARITY_FILE = "kg_similarity.npz"
KG_TUNING_RESULTS_FILE = "kg_tuning_results.json"

# Hybrid
HYBRID_BEST_PARAMS_FILE = "hybrid_best_params.json"
HYBRID_GRID_RESULTS_FILE = "hybrid_grid_results.json"
HYBRID_MATRIX_FILE = "hybrid_matrix.npz"

# Evaluation
METRICS_FILE = "metrics.json"
SUMMARY_FILE = "summary.md"