import os
# Lấy danh sách stopwords tiếng Anh chuẩn của scikit-learn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Danh sách stopwords thực tế bạn đã chốt ở bước trước
YOUR_RETAIL_STOPWORDS = {
    'organic', 'natural', 'original', 'premium', 'free', 'low', 'sodium',
    'classic', 'traditional', 'authentic', 'delicious', 'simply', 'pure',
    'ultra', 'super', 'value', 'wild', 'fresh', 'style', 'fat', 'lowfat',
    'total', 'light', 'diet', 'with', 'and', 'for', 'whole', 'sweet',
    'gluten', 'dark', 'roasted', 'creamy', 'soft', 'frozen', 'mini',
    'medium', 'all', 'extra', 'blend', 'greek', 'ground', 'flavor', 'scent',
    'liquid', 'hot'
}

# 2. DANH SÁCH BẢO VỆ (Tuyệt đối KHÔNG ĐƯỢC XÓA vì là tên sản phẩm trong dữ liệu thực tế)
PROTECTED_RETAIL_WORDS = {
    'chicken', 'cheese', 'chocolate', 'sauce', 'cream', 'yogurt', 'milk', 
    'tea', 'butter', 'vanilla', 'white', 'bar', 'rice', 'fruit', 'juice', 
    'food', 'oil', 'coconut', 'chips', 'coffee', 'strawberry', 'apple', 
    'lemon', 'green', 'salt', 'cheddar', 'black', 'water', 'honey', 'soup', 
    'ice', 'bread', 'baby', 'almond', 'sugar', 'red', 'grain', 'beef', 
    'garlic', 'pasta', 'orange', 'drink', 'potato', 'cookies', 'protein', 
    'peanut', 'bars', 'corn', 'sea', 'mint', 'beans', 'wheat', 'pizza', 
    'brown', 'turkey', 'crackers', 'vegetable', 'cherry', 'cereal', 'cat', 
    'dressing', 'blueberry', 'dog', 'body', 'cinnamon', 'italian', 'tomato', 
    'caramel', 'shampoo', 'snack', 'granola', 'bean', 'raspberry', 'ginger', 
    'sausage', 'thin'
}

def generate_ultimate_stopwords():
    # Chuyển ENGLISH_STOP_WORDS của sklearn sang dạng set từ thường
    sklearn_stopwords = {word.lower() for word in ENGLISH_STOP_WORDS}
    
    # Gộp danh sách của bạn và sklearn lại làm một
    combined_stopwords = YOUR_RETAIL_STOPWORDS.union(sklearn_stopwords)
    
    # LUẬT BẢO VỆ: Loại bỏ các từ cốt lõi ra khỏi danh sách xóa
    final_stopwords = combined_stopwords - PROTECTED_RETAIL_WORDS
    
    # Ghi đè file cấu hình data/retail_stopwords.txt
    os.makedirs('data', exist_ok=True)
    with open('data/retail_stopwords.txt', 'w', encoding='utf-8') as f:
        for word in sorted(final_stopwords):
            f.write(f"{word}\n")
            
    print(f"[XONG] Đã gộp thành công danh sách tự chọn và Sklearn.")
    print(f"Tổng số từ khóa hủy diệt đã lọc bảo vệ: {len(final_stopwords)} từ.")
    print("Mời bạn chạy lại hệ thống đánh giá chính!")

if __name__ == "__main__":
    generate_ultimate_stopwords()