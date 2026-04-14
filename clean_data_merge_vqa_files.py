import json
import os

def merge_and_sort_vqa(input_folder, output_file):
    merged_list = []
    
    # 1. Kiểm tra folder đầu vào
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' không tồn tại!")
        return

    # 2. Lấy danh sách các file json
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    print(f"Tìm thấy {len(json_files)} file JSON. Đang đọc dữ liệu...")

    for file_name in sorted(json_files):
        file_path = os.path.join(input_folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for image_id, content in data.items():
                    # Lấy ra list QA tùy theo format (dict có key "qa" hoặc list trực tiếp)
                    if isinstance(content, dict) and "qa" in content:
                        qa_content = content["qa"]
                    else:
                        qa_content = content
                    
                    # Lưu vào list dưới dạng tuple (image_id, qa_content) để sort
                    merged_list.append((image_id, qa_content))
                        
        except Exception as e:
            print(f"Lỗi khi đọc file {file_name}: {e}")

    # 3. Sắp xếp theo số lượng cặp QA (len của list QA)
    # reverse=True nếu bạn muốn nhiều QA nhất lên đầu
    # reverse=False nếu bạn muốn ít QA nhất lên đầu
    print("Đang sắp xếp theo số lượng cặp QA...")
    merged_list.sort(key=lambda x: len(x[1]))

    # 4. Chuyển list đã sort thành dictionary để lưu JSON
    final_dict = dict(merged_list)

    # 5. Lưu file
    print(f"Tổng cộng có {len(final_dict)} image_id. Đang lưu vào {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_dict, f, indent=4, ensure_ascii=False)
        print("Hợp nhất và sắp xếp thành công!")
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")

# --- Chạy script ---
if __name__ == "__main__":
    INPUT_FOLDER = 'image_vqa_folder' 
    OUTPUT_FILE = 'image_vqa.json'
    
    merge_and_sort_vqa(INPUT_FOLDER, OUTPUT_FILE)