import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Đường dẫn tới các file của bạn
PATH_JSON = r"C:\Users\b2h16\OneDrive\Máy tính\recommender-system\models\cb_filter\product_id_to_idx.json"
PATH_NPZ = r"C:\Users\b2h16\OneDrive\Máy tính\recommender-system\models\cb_filter\product_vectors.npz"

def load_data():
    print("🔄 Đang tải dữ liệu...")
    # Load file mapping ID -> Index
    with open(PATH_JSON, 'json', encoding='utf-8') as f:
        product_id_to_idx = json.load(f)
    
    # Đảo ngược mapping để từ Index tìm lại được ID sản phẩm
    idx_to_product_id = {int(v): k for k, v in product_id_to_idx.items()}
    
    # Load ma trận vector sản phẩm (giả sử key trong npz là 'vectors' hoặc 'arr_0')
    npz_file = np.load(PATH_NPZ)
    # Lấy key đầu tiên nếu không biết chính xác tên key lưu trữ
    key = npz_file.files[0]
    product_vectors = npz_file[key]
    
    # Nếu là ma trận thưa từ scipy được lưu dạng npz, cần convert sang dense/csr nếu cần
    # Thường nếu dùng np.savez thì nó là numpy array thông thường
    if hasattr(product_vectors, "toarray"):
        product_vectors = product_vectors.toarray()
        
    print(f"✅ Tải thành công! Số lượng sản phẩm: {len(product_id_to_idx)}")
    print(f"📐 Kích thước ma trận vector: {product_vectors.shape}")
    return product_id_to_idx, idx_to_product_id, product_vectors

def recommend_sanity_check(target_product_id, idx_to_product_id, product_id_to_idx, product_vectors, top_k=5):
    if target_product_id not in product_id_to_idx:
        print(f"❌ Không tìm thấy Product ID: {target_product_id} trong từ điển.")
        return
    
    # Lấy index và vector của sản phẩm mục tiêu
    target_idx = product_id_to_idx[target_product_id]
    target_vector = product_vectors[target_idx].reshape(1, -1)
    
    # Tính toán Cosine Similarity giữa sản phẩm này với TOÀN BỘ sản phẩm khác
    # Kết quả trả về một mảng chứa điểm số tương đồng
    similarity_scores = cosine_similarity(target_vector, product_vectors).flatten()
    
    # Sắp xếp lấy index của các sản phẩm có điểm cao nhất (giảm dần)
    # Loại bỏ chính nó (sản phẩm mục tiêu luôn có similarity = 1.0 với chính nó)
    related_indices = np.argsort(similarity_scores)[::-1]
    related_indices = [idx for idx in related_indices if idx != target_idx][:top_k]
    
    print("\n" + "="*50)
    print(f"🎯 Sản phẩm gốc (Target): ID {target_product_id} (Index: {target_idx})")
    print("="*50)
    print(f"🤖 Top {top_k} sản phẩm tương đồng nhất được gợi ý:")
    
    for i, idx in enumerate(related_indices, 1):
        p_id = idx_to_product_id[idx]
        score = similarity_scores[idx]
        print(f"{i}. Product ID: {p_id} | Điểm Cosine: {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    try:
        product_id_to_idx, idx_to_product_id, product_vectors = load_data()
        
        # Lấy thử một ID sản phẩm bất kỳ từ dữ liệu để test
        sample_product_id = list(product_id_to_idx.keys())[0] 
        
        # Chạy thử nghiệm gợi ý
        recommend_sanity_check(
            target_product_id=sample_product_id, 
            idx_to_product_id=idx_to_product_id, 
            product_id_to_idx=product_id_to_idx, 
            product_vectors=product_vectors, 
            top_k=5
        )
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
        print("💡 Gợi ý: Hãy kiểm tra lại định dạng lưu trữ bên trong file .npz xem key chính xác là gì.")