"""
Metrics evaluation cho bundle recommendation system.
Tính các chỉ số Precision@K, Recall@K, F1@K, Hit@K
dựa trên ground truth từ LLM survey (Mục 5.4 docs/models.md).
"""
import numpy as np
from typing import Dict, List, Tuple


def compute_precision_at_k(
    ground_truth: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Precision@K = số lượng gợi ý đúng (complementary) trong top-K / K
    
    Args:
        ground_truth: dict {product_A_id: [list product_B_id complementary]}
        predictions: dict {product_A_id: [list product_B_id top-K gợi ý]}
        k: int
    
    Returns:
        float: Precision@K trung bình trên tất cả sản phẩm đầu vào
    """
    precisions = []
    
    for pid_a, true_complementary in ground_truth.items():
        if pid_a not in predictions:
            continue
        
        pred_top_k = predictions[pid_a][:k]
        if len(pred_top_k) == 0:
            precisions.append(0.0)
            continue
        
        # Đếm số lượng gợi ý đúng trong top-K
        true_set = set(true_complementary)
        hits = sum(1 for pid_b in pred_top_k if pid_b in true_set)
        
        precisions.append(hits / k)
    
    return float(np.mean(precisions)) if precisions else 0.0


def compute_recall_at_k(
    ground_truth: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Recall@K = số lượng gợi ý đúng trong top-K / tổng số complementary trong ground truth
    
    Args:
        ground_truth: dict {product_A_id: [list product_B_id complementary]}
        predictions: dict {product_A_id: [list product_B_id top-K gợi ý]}
        k: int
    
    Returns:
        float: Recall@K trung bình trên tất cả sản phẩm đầu vào
    """
    recalls = []
    
    for pid_a, true_complementary in ground_truth.items():
        if pid_a not in predictions:
            continue
        
        pred_top_k = predictions[pid_a][:k]
        true_set = set(true_complementary)
        
        if len(true_set) == 0:
            # Không có ground truth complementary → không tính recall
            continue
        
        hits = sum(1 for pid_b in pred_top_k if pid_b in true_set)
        recalls.append(hits / len(true_set))
    
    return float(np.mean(recalls)) if recalls else 0.0


def compute_f1_at_k(
    precision: float,
    recall: float
) -> float:
    """
    F1@K = 2 × (Precision@K × Recall@K) / (Precision@K + Recall@K)
    
    Args:
        precision: float — Precision@K
        recall: float — Recall@K
    
    Returns:
        float: F1@K
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_hit_at_k(
    ground_truth: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Hit@K = tỷ lệ product_A có ít nhất 1 gợi ý đúng trong top-K
    
    Args:
        ground_truth: dict {product_A_id: [list product_B_id complementary]}
        predictions: dict {product_A_id: [list product_B_id top-K gợi ý]}
        k: int
    
    Returns:
        float: Hit@K (tỷ lệ 0.0 - 1.0)
    """
    hits = 0
    total = 0
    
    for pid_a, true_complementary in ground_truth.items():
        if pid_a not in predictions:
            continue
        
        pred_top_k = predictions[pid_a][:k]
        true_set = set(true_complementary)
        
        total += 1
        if any(pid_b in true_set for pid_b in pred_top_k):
            hits += 1
    
    return hits / total if total > 0 else 0.0


def compute_all_metrics(
    ground_truth: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
    k: int = 10
) -> Dict[str, float]:
    """
    Tính tất cả các metrics cho 1 model.
    
    Args:
        ground_truth: dict {product_A_id: [list product_B_id complementary]}
        predictions: dict {product_A_id: [list product_B_id top-K gợi ý]}
        k: int
    
    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float, 'hit': float}
    """
    precision = compute_precision_at_k(ground_truth, predictions, k)
    recall = compute_recall_at_k(ground_truth, predictions, k)
    f1 = compute_f1_at_k(precision, recall)
    hit = compute_hit_at_k(ground_truth, predictions, k)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hit': hit,
        'k': k,
    }