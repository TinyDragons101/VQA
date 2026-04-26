import json

def check_vqa_validity(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    total_ids = len(data)
    total_pairs = 0
    invalid_pairs = 0
    invalid_details = []

    for img_id, qa_list in data.items():
        for index, pair in enumerate(qa_list):
            total_pairs += 1
            # Kiểm tra nếu không phải list hoặc số lượng phần tử khác 2
            if not isinstance(pair, list) or len(pair) != 2:
                invalid_pairs += 1
                invalid_details.append({
                    "id": img_id,
                    "index": index,
                    "content": pair
                })

    # Xuất kết quả
    print("--- THỐNG KÊ DỮ LIỆU VQA ---")
    print(f"Tổng số ID ảnh: {total_ids}")
    print(f"Tổng số cặp QA đã quét: {total_pairs}")
    print(f"Số cặp KHÔNG hợp lệ: {invalid_pairs}")
    
    if invalid_pairs > 0:
        print("\n--- CHI TIẾT LỖI ---")
        for detail in invalid_details:
            print(f"ID: {detail['id']} | Vị trí: {detail['index']} | Nội dung: {detail['content']}")
    else:
        print("\nChúc mừng! Tất cả các cặp QA đều hợp lệ (đúng 2 phần tử).")

# Thay 'image_vqa.json' bằng đường dẫn file thực tế của bạn
check_vqa_validity('./image_vqa.json')