# survey_cb.py
"""
Khảo sát CB similarity distribution - chạy trực tiếp bằng python survey_cb.py
"""

import sys
import json
import numpy as np
import pandas as pd
import random
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("="*60)
    print("CB SIMILARITY SURVEY TOOL")
    print("="*60)
    
    # Load model
    print("\nLoading CB model...")
    product_vectors = sparse.load_npz('models/cb_filter/product_vectors.npz')
    with open('models/cb_filter/product_id_to_idx.json', 'r') as f:
        product_id_to_idx = json.load(f)
    
    product_ids = list(product_id_to_idx.keys())
    products_df = pd.read_csv('data/products.csv')
    
    print(f"  Products loaded: {len(product_ids)}")
    print(f"  Vector shape: {product_vectors.shape}")
    
    # Define similarity function
    def cb_similarity(vectors, idx_a, idxs_b):
        vec_a = vectors[idx_a]
        vecs_b = vectors[idxs_b]
        return cosine_similarity(vec_a.reshape(1, -1), vecs_b).flatten()
    
    # Sample random pairs
    print("\nSampling 5000 random pairs...")
    random.seed(42)
    sims = []
    
    for _ in range(5000):
        a = random.choice(product_ids)
        b = random.choice(product_ids)
        if a == b:
            continue
        idx_a = product_id_to_idx[a]
        idx_b = product_id_to_idx[b]
        sim = cb_similarity(product_vectors, idx_a, [idx_b])[0]
        sims.append(sim)
    
    sims = np.array(sims)
    
    # Statistics
    print("\n" + "="*60)
    print("SIMILARITY DISTRIBUTION STATISTICS")
    print("="*60)
    print(f"Number of pairs sampled: {len(sims)}")
    print(f"Mean:     {sims.mean():.6f}")
    print(f"Std:      {sims.std():.6f}")
    print(f"Min:      {sims.min():.6f}")
    print(f"25th:     {np.percentile(sims, 25):.6f}")
    print(f"50th:     {np.percentile(sims, 50):.6f}")
    print(f"75th:     {np.percentile(sims, 75):.6f}")
    print(f"90th:     {np.percentile(sims, 90):.6f}")
    print(f"95th:     {np.percentile(sims, 95):.6f}")
    print(f"99th:     {np.percentile(sims, 99):.6f}")
    print(f"Max:      {sims.max():.6f}")
    
    # Threshold analysis - từ 0 đến 0.95, bước 0.05
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS (percentage of pairs filtered)")
    print("="*60)
    print(f"{'Threshold':<12} {'% Filtered (≥ threshold)':<25} {'% Kept (< threshold)':<22} {'Interpretation'}")
    print("-"*80)
    
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    for th in thresholds:
        pct_filtered = (sims >= th).mean() * 100
        pct_kept = 100 - pct_filtered
        
        if th == 0:
            interp = "Keep everything"
        elif th >= 0.9:
            interp = "Very strict - only exact duplicates"
        elif th >= 0.8:
            interp = "Strict - good for substitute detection"
        elif th >= 0.7:
            interp = "Moderate"
        elif th >= 0.5:
            interp = "Loose"
        else:
            interp = "Very loose - keeps most"
        
        print(f"{th:<12} {pct_filtered:<25.2f}% {pct_kept:<22.2f}% {interp}")
    
    # Histogram-like distribution (bins)
    print("\n" + "="*60)
    print("SIMILARITY DISTRIBUTION (binned)")
    print("="*60)
    
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    for i in range(len(bins)-1):
        low = bins[i]
        high = bins[i+1]
        count = np.sum((sims >= low) & (sims < high))
        pct = count / len(sims) * 100
        bar = "█" * int(pct / 2)
        print(f"{low:.2f} - {high:.2f}: {pct:5.2f}% {bar}")
    
    # Top similar pairs
    print("\n" + "="*60)
    print("TOP 20 MOST SIMILAR PAIRS (potential substitutes)")
    print("="*60)
    
    # Collect top similar pairs
    pairs = []
    for _ in range(2000):
        a = random.choice(product_ids)
        b = random.choice(product_ids)
        if a == b:
            continue
        sim = cb_similarity(product_vectors, product_id_to_idx[a], [product_id_to_idx[b]])[0]
        pairs.append((a, b, sim))
    
    top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:20]
    name_dict = dict(zip(products_df['product_id'], products_df['product_name']))
    
    print(f"{'Rank':<5} {'ID_A':<8} {'ID_B':<8} {'Similarity':<12} {'Product A':<35} {'Product B'}")
    print("-"*90)
    for rank, (a, b, s) in enumerate(top_pairs, 1):
        name_a = name_dict.get(a, 'Unknown')[:32]
        name_b = name_dict.get(b, 'Unknown')[:32]
        print(f"{rank:<5} {a:<8} {b:<8} {s:<12.4f} {name_a:<35} {name_b}")
    
    # Analyze specific product if requested
    print("\n" + "="*60)
    print("ANALYZE A SPECIFIC PRODUCT")
    print("="*60)
    print("Enter a product_id to see its most similar products (or press Enter to skip)")
    
    try:
        pid_input = input("Product ID: ").strip()
        if pid_input:
            pid = int(pid_input)
            if pid in product_id_to_idx:
                idx_a = product_id_to_idx[pid]
                name_a = name_dict.get(pid, 'Unknown')
                print(f"\nProduct: {pid} - {name_a}")
                
                # Get all other products
                other_ids = [p for p in product_ids if p != pid]
                other_idxs = [product_id_to_idx[p] for p in other_ids]
                sims_pid = cb_similarity(product_vectors, idx_a, other_idxs)
                
                # Top 20 most similar
                top_indices = np.argsort(sims_pid)[::-1][:20]
                
                print(f"\nTOP 20 MOST SIMILAR TO {pid}:")
                print(f"{'Rank':<5} {'ID':<8} {'Similarity':<12} {'Product Name'}")
                print("-"*70)
                for rank, idx in enumerate(top_indices, 1):
                    pid_other = other_ids[idx]
                    sim_val = sims_pid[idx]
                    name_other = name_dict.get(pid_other, 'Unknown')[:45]
                    print(f"{rank:<5} {pid_other:<8} {sim_val:<12.4f} {name_other}")
                
                # Count by threshold
                print(f"\nTHRESHOLD COUNTS for product {pid}:")
                for th in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                    count = np.sum(sims_pid >= th)
                    print(f"  ≥ {th}: {count:5d} products ({count/len(sims_pid)*100:5.2f}%)")
            else:
                print(f"Product {pid} not found in CB model")
    except (ValueError, KeyboardInterrupt):
        pass
    
    print("\n" + "="*60)
    print("SURVEY COMPLETE")
    print("="*60)
    
    # Recommendation
    print("\nRECOMMENDED THRESHOLD:")
    print(f"  - 99th percentile of random pairs: {np.percentile(sims, 99):.4f}")
    print(f"  - 95th percentile of random pairs: {np.percentile(sims, 95):.4f}")
    print(f"  - 90th percentile of random pairs: {np.percentile(sims, 90):.4f}")
    print("\n  Suggestion: Use threshold = 0.8 or 0.85")
    print(f"    - 0.8: Filters {(sims >= 0.8).mean() * 100:.2f}% of random pairs")
    print(f"    - 0.85: Filters {(sims >= 0.85).mean() * 100:.2f}% of random pairs")
    print(f"    - 0.9: Filters {(sims >= 0.9).mean() * 100:.2f}% of random pairs")

if __name__ == "__main__":
    main()