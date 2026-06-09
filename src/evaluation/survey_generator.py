"""
Sinh mẫu khảo sát (survey samples) cho Association Rules.
Phương pháp: Mục 5.2.2 docs/models.md
  - Top-5 (50%): lấy product_B từ top-5 recommend của model
  - Noise (50%): lấy product_B ngẫu nhiên từ sản phẩm KHÔNG nằm trong top-K
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random
import numpy as np
import pandas as pd

from src.config import DATA_DIR, RANDOM_SEED


def generate_survey_samples(
    model,
    products_df: pd.DataFrame,
    model_name: str = "assoc_rules",
    n_samples: int = 100,
    top_k: int = 100,
    seed: int = None
) -> pd.DataFrame:
    """
    Sinh mẫu khảo sát cho 1 model.
    
    Args:
        model: model có method recommend(product_id, top_k) -> list (product_id, score)
        products_df: DataFrame [product_id, product_name, ...]
        model_name: str, tên model
        n_samples: int, số lượng product_A cần lấy mẫu
        top_k: int, số lượng top-K recommend để lấy noise candidates
        seed: int, random seed
    
    Returns:
        DataFrame columns: [product_A_id, product_B_id, model_name, source]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    all_product_ids = sorted(products_df['product_id'].unique())
    
    # Chọn n_samples product_A ngẫu nhiên
    sampled_a = random.sample(all_product_ids, min(n_samples, len(all_product_ids)))
    
    records = []
    
    for pid_a in sampled_a:
        # Lấy top-K recommend từ model
        recs = model.recommend(pid_a, top_k=top_k)
        rec_ids = [r[0] for r in recs]
        rec_set = set(rec_ids)
        
        # Top-5: lấy 5 product_B từ top-5
        top5_ids = rec_ids[:5] if len(rec_ids) >= 5 else rec_ids
        for pid_b in top5_ids:
            records.append({
                'product_A_id': pid_a,
                'product_B_id': pid_b,
                'model_name': model_name,
                'source': 'top5',
            })
        
        # Noise: lấy 5 product_B ngẫu nhiên KHÔNG nằm trong top-K
        noise_candidates = [pid for pid in all_product_ids
                           if pid != pid_a and pid not in rec_set]
        if len(noise_candidates) >= 5:
            noise_ids = random.sample(noise_candidates, 5)
        else:
            noise_ids = noise_candidates
        
        for pid_b in noise_ids:
            records.append({
                'product_A_id': pid_a,
                'product_B_id': pid_b,
                'model_name': model_name,
                'source': 'noise',
            })
    
    df = pd.DataFrame(records)
    return df


def generate_assoc_rules_survey(
    model,
    products_df: pd.DataFrame,
    model_name: str = "assoc_rules",
    n_samples: int = 100,
    top_k: int = 100,
    seed: int = None,
    output_dir: str = None
) -> str:
    """
    Sinh survey samples cho Association Rules và lưu ra CSV.
    
    Args:
        model: AssocRulesModel instance
        products_df: DataFrame products
        model_name: str
        n_samples: int, số product_A
        top_k: int, top-K để lấy noise candidates
        seed: int
        output_dir: str, đường dẫn thư mục output (default: data/survey/)
    
    Returns:
        str: đường dẫn file CSV đã lưu
    """
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "survey")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = generate_survey_samples(
        model=model,
        products_df=products_df,
        model_name=model_name,
        n_samples=n_samples,
        top_k=top_k,
        seed=seed,
    )
    
    output_path = os.path.join(output_dir, f"{model_name}_samples.csv")
    df.to_csv(output_path, index=False)
    print(f"Survey samples saved: {output_path}")
    print(f"  Tổng số mẫu: {len(df)}")
    print(f"  Top-5: {len(df[df['source'] == 'top5'])}")
    print(f"  Noise: {len(df[df['source'] == 'noise'])}")
    
    return output_path