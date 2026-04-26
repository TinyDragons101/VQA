import json
from collections import Counter

def analyze_qa_difficulty(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    total_image_ids = len(data)
    total_qa_pairs = 0
    difficulty_counts = Counter()
    missing_difficulty = 0

    # Duyệt qua từng image_id
    for img_id, qa_list in data.items():
        for qa_pair in qa_list:
            total_qa_pairs += 1
            
            # Giả sử cấu trúc mỗi cặp QA là [question, answer, difficulty]
            # Kiểm tra xem có đủ 3 phần tử không
            if len(qa_pair) < 3:
                missing_difficulty += 1
                continue
            
            difficulty = qa_pair[2]
            
            if difficulty and difficulty.isdigit():
                difficulty_counts[difficulty] += 1
            else:
                missing_difficulty += 1

    # In kết quả
    print("-" * 30)
    print(f"TỔNG KẾT DỮ LIỆU")
    print("-" * 30)
    print(f"Tổng số Image ID: {total_image_ids}")
    print(f"Tổng số cặp QA   : {total_qa_pairs}")
    print(f"Cặp QA thiếu độ khó: {missing_difficulty}")
    
    print("\nThống kê theo độ khó (1-6):")
    for d in range(1, 7):
        count = difficulty_counts.get(str(d), 0)
        print(f"Độ khó {d}: {count} cặp")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="./image_vqa_with_difficulty_cleaned.json")
    
    args = parser.parse_args()
    analyze_qa_difficulty(args.file_path)