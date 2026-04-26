import json
import os

def process_json_files(input_file, output_file):
    # 1. Kiểm tra nếu file đầu vào tồn tại
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    # 2. Đọc dữ liệu từ file input
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Lỗi cú pháp JSON trong file input: {e}")
            return

    fixed_data = {}

    # 3. Duyệt qua từng ID ảnh
    for img_id, content in data.items():
        # Kiểm tra nếu content là một danh sách và phần tử đầu tiên là String
        # Điều này chứng tỏ dữ liệu đang bị "phẳng" (flattened)
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], str):
            new_pairs = []
            # Gom 2 phần tử liên tiếp thành 1 cặp [Câu hỏi, Trả lời]
            for i in range(0, len(content) - 1, 2):
                new_pairs.append([content[i], content[i+1]])
            fixed_data[img_id] = new_pairs
        else:
            # Nếu đã đúng định dạng [[q, a]] hoặc rỗng thì giữ nguyên
            fixed_data[img_id] = content

    # 4. Ghi dữ liệu đã xử lý ra file output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Xử lý thành công! Dữ liệu sạch đã được lưu tại: {output_file}")

# --- Cấu hình tên file ---
INPUT_NAME = "image_vqa.json"   # Tên file lỗi của bạn
OUTPUT_NAME = "image_vqa.json" # Tên file kết quả

if __name__ == "__main__":
    process_json_files(INPUT_NAME, OUTPUT_NAME)