import asyncio
import os
import edge_tts

# 1. Đường dẫn file text của bạn
INPUT_TEXT_FILE = r"C:\Users\b2h16\Downloads\Tài liệu không có tiêu đề (1).txt"

# 2. Đường dẫn file âm thanh đầu ra (sẽ lưu cùng thư mục với file text)
OUTPUT_AUDIO_FILE = r"C:\Users\b2h16\Downloads\output_voice (1).mp3"

# 3. Chọn giọng đọc tiếng Việt (Hoài My hoặc Nam Minh)
# Giọng nữ: "vi-VN-HoaiMyNeural"
# Giọng nam: "vi-VN-NamMinhNeural"
VOICE = "vi-VN-HoaiMyNeural"


async def text_to_speech():
    # Kiểm tra xem file txt có tồn tại không
    if not os.path.exists(INPUT_TEXT_FILE):
        print(f"Không tìm thấy file tại đường dẫn: {INPUT_TEXT_FILE}")
        return

    print("Đang đọc nội dung file text...")
    # Đọc nội dung file với mã hóa utf-8 để tránh lỗi font tiếng Việt
    with open(INPUT_TEXT_FILE, "r", encoding="utf-8") as f:
        text_content = f.read()

    if not text_content.strip():
        print("File text rỗng, không có gì để chuyển đổi!")
        return

    print("Đang tiến hành chuyển đổi thành giọng nói...")
    communicate = edge_tts.Communicate(text_content, VOICE)

    # Lưu thành file mp3
    await communicate.save(OUTPUT_AUDIO_FILE)
    print(f"Hoàn thành! File âm thanh đã được lưu tại: {OUTPUT_AUDIO_FILE}")


if __name__ == "__main__":
    # Chạy hàm async
    asyncio.run(text_to_speech())