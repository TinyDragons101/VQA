import json
import os
import argparse

def clean_vqa_data(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    initial_count = len(data)
    cleaned_data = {}
    removed_count = 0

    print(f"[*] Đang kiểm tra {initial_count} Image IDs...")

    for img_id, qa_list in data.items():
        is_valid = True
        
        # Kiểm tra từng cặp QA
        for qa_pair in qa_list:
            # Điều kiện hợp lệ: phải là list và có ít nhất 3 phần tử [Q, A, Diff]
            if not isinstance(qa_pair, list) or len(qa_pair) < 3:
                is_valid = False
                break
            
            # Kiểm tra thêm nếu difficulty là chuỗi rỗng hoặc None
            if not str(qa_pair[2]).strip() or len(str(qa_pair[2]).strip()) != 1:
                is_valid = False
                break

        if is_valid:
            cleaned_data[img_id] = qa_list
        else:
            print("Removeed Image ID:", img_id)
            removed_count += 1

    # Ghi file mới
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        print("-" * 40)
        print(f"KẾT QUẢ DỌN DẸP:")
        print(f"- Tổng số ban đầu  : {initial_count}")
        print(f"- Số ID bị loại bỏ : {removed_count}")
        print(f"- Số ID còn lại    : {len(cleaned_data)}")
        print(f"- File đã lưu tại  : {output_path}")
        print("-" * 40)
        
    except Exception as e:
        print(f"Lỗi khi ghi file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loại bỏ các image_id có QA thiếu độ khó.")
    parser.add_argument("--input", type=str, default="./image_vqa_with_difficulty_cleaned.json")
    parser.add_argument("--output", type=str, default="./image_vqa_with_difficulty_cleaned.json")
    
    args = parser.parse_args()
    clean_vqa_data(args.input, args.output)