import urllib.request

print("🔄 Đang tải và nạp bộ từ dừng từ GitHub bạn cung cấp...")

# 2. Đường dẫn RAW đến file vietnamese-stopwords-dash.txt trên GitHub của bạn
url_vi = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords-dash.txt"

try:
    with urllib.request.urlopen(url_vi) as response:
        # Đọc dữ liệu và giải mã sang chuỗi UTF-8
        raw_lines = response.read().decode('utf-8').splitlines()
    
    vi_stop_words_raw = []
    for line in raw_lines:
        word = line.strip()
        if word:
            # Chuyển dấu gạch ngang (-) thành khoảng trắng ( ) để khớp với Tokenizer của scikit-learn
            word_clean = word.replace('-', ' ')
            vi_stop_words_raw.append(word_clean)

except Exception as e:
    print(f"❌ Lỗi kết nối mạng khi tải từ GitHub: {e}")
    vi_stop_words_raw = []


# --- ĐOẠN ĐỌC FILE TIẾNG ANH CỦA BẠN ---
english_file_path = r"C:\Users\b2h16\OneDrive\Máy tính\recommender-system\english_stopwords.txt"
try:
    with open(english_file_path, "r", encoding="utf-8") as f:
        en_stop_words = [line.strip() for line in f.readlines() if line.strip()]
except Exception as e:
    print(f"❌ Lỗi khi đọc file Tiếng Anh: {e}")
    en_stop_words = []


# 4. Gộp hai bộ từ lại thành Bộ Từ Dừng Song Ngữ duy nhất cho model lai
hybrid_stop_words = en_stop_words + vi_stop_words_raw


# --- ĐOẠN GHI KẾT QUẢ RA FILE MỚI ---
output_file_path = r"C:\Users\b2h16\OneDrive\Máy tính\recommender-system\hybrid_stopwords.txt"
try:
    with open(output_file_path, "w", encoding="utf-8") as f:
        for word in hybrid_stop_words:
            f.write(f"{word}\n")
    print(f"💾 Đã ghi toàn bộ từ dừng lai vào file: {output_file_path}")
except Exception as e:
    print(f"❌ Lỗi khi ghi file kết quả: {e}")


print(f"✅ THÀNH CÔNG!")
print(f"Tổng số từ dừng lai Anh - Việt đưa vào mô hình: {len(hybrid_stop_words)} từ.")

print("\nVí dụ 10 từ dừng Tiếng Việt lấy từ file GitHub của bạn (sau khi đổi dấu gạch ngang):")
print(vi_stop_words_raw[20:30])