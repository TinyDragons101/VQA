import json
import os

def filter_vqa_data(input_file, output_file):
    print(f"--- Đang lọc dữ liệu: {input_file} ---")
    
    if not os.path.exists(input_file):
        print("Lỗi: File đầu vào không tồn tại.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = {}
    stats = {
        "removed_single_element": 0,
        "removed_low_qa_count": 0,
        "kept": 0
    }

    for img_id, content in data.items():
        qa_list = content.get("qa", [])
        
        # ĐIỀU KIỆN 1: Kiểm tra xem có cặp QA nào chỉ có 1 phần tử không
        # (Cấu trúc đúng phải là [Q, A] - tức 2 phần tử)
        has_invalid_pair = any(len(pair) < 2 for pair in qa_list)
        
        if has_invalid_pair:
            stats["removed_single_element"] += 1
            continue
            
        # ĐIỀU KIỆN 2: Chỉ giữ lại những image_id có từ 3 cặp QA trở lên
        if len(qa_list) <= 3:
            stats["removed_low_qa_count"] += 1
            continue

        # Nếu vượt qua cả 2 điều kiện thì giữ lại
        filtered_data[img_id] = content
        stats["kept"] += 1

    # Lưu file kết quả
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"--- BÁO CÁO LỌC DỮ LIỆU ---")
    print(f"1. Số ảnh bị xóa do QA thiếu cặp (chỉ có 1 phần tử): {stats['removed_single_element']}")
    print(f"2. Số ảnh bị xóa do có quá ít QA (<= 2 cặp): {stats['removed_low_qa_count']}")
    print(f"3. Số ảnh đạt tiêu chuẩn giữ lại: {stats['kept']}")
    print(f"--- Đã lưu file sạch tại: {output_file} ---")

if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn tại đây
    INPUT_FILE = './image_vqa_folder/image_vqa_01.json'
    OUTPUT_FILE = './image_vqa_folder/image_vqa_01.json'
    
    filter_vqa_data(INPUT_FILE, OUTPUT_FILE)