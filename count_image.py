import json
from collections import Counter

def count_duplicate_keys_in_file(file_path):
    all_keys = []

    # Hàm này sẽ được gọi mỗi khi thư viện json tìm thấy một cặp key-value
    def collect_keys(pairs):
        for k, v in pairs:
            all_keys.append(k)
        return dict(pairs) # Trả về dict bình thường để không làm hỏng cấu trúc

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Dùng object_pairs_hook để "bắt" tất cả các key trước khi chúng bị ghi đè
            json.load(f, object_pairs_hook=collect_keys)

        # Thống kê
        counts = Counter(all_keys)
        duplicates = {k: v for k, v in counts.items() if v > 1}

        print(f"--- THỐNG KÊ FILE: {file_path} ---")
        print(f"Tổng số lượt xuất hiện của các ID: {len(all_keys)}")
        print(f"Số ID duy nhất: {len(counts)}")
        print(f"Số ID bị trùng lặp bên trong file: {len(duplicates)}")

        if duplicates:
            print("\n--- CHI TIẾT CÁC ID TRÙNG ---")
            for img_id, count in list(duplicates.items())[:10]: # Hiện 10 cái đầu
                print(f"ID: {img_id} | Xuất hiện: {count} lần")
        else:
            print("\nChúc mừng! Không có ID nào bị trùng lặp trong file này.")

    except Exception as e:
        print(f"Lỗi: {e}")

# Sử dụng
count_duplicate_keys_in_file('image_vqa.json')