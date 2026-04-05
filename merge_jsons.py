import json
import os

# Cấu hình tên file
target_ids_file = "image_caption_updated.json"
file_input_1 = "image_questions.json"
file_input_2 = "image_questions_old.json"
output_combined = "combined_captions.json"

def merge_json_files():
    # 1. Lấy danh sách image_id mục tiêu từ image_caption_updated.json
    if not os.path.exists(target_ids_file):
        print(f"[!] Không tìm thấy file gốc {target_ids_file}")
        return

    with open(target_ids_file, "r", encoding="utf-8") as f:
        target_data = json.load(f)
        # Lấy set các key (image_id) để tra cứu nhanh
        allowed_ids = set(target_data.keys())
    
    print(f"[*] Đã nạp {len(allowed_ids)} ID mục tiêu từ {target_ids_file}")

    combined_data = {}
    files_to_process = [file_input_1, file_input_2]

    # 2. Quét qua 2 file JSON cần gom
    for file_path in files_to_process:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    added_from_this_file = 0
                    for img_id, content in data.items():
                        # Kiểm tra xem ID có nằm trong danh sách cho phép không
                        if img_id in allowed_ids:
                            # Nếu ID đã có trong combined_data, bạn có thể chọn ghi đè 
                            # hoặc giữ nguyên. Ở đây tôi chọn ghi đè/cập nhật.
                            combined_data[img_id] = content
                            added_from_this_file += 1
                    
                    print(f"[+] Đã lọc và lấy {added_from_this_file} entries từ {file_path}")
            except Exception as e:
                print(f"[!] Lỗi khi xử lý {file_path}: {e}")
        else:
            print(f"[!] File {file_path} không tồn tại, bỏ qua.")

    # 3. Xuất file JSON kết quả
    with open(output_combined, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print(f"[OK] Hoàn tất!")
    print(f"- Tổng số entries sau khi lọc và gom: {len(combined_data)}")
    print(f"- File kết quả: {output_combined}")

if __name__ == "__main__":
    merge_json_files()