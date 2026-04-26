import json
import os

def fix_vqa_only(input_file, output_file):
    print(f"--- Đang khởi động xử lý QA: {input_file} ---")
    
    if not os.path.exists(input_file):
        print(f"Lỗi: File {input_file} không tồn tại.")
        return

    # 1. Đọc file VQA
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Lỗi cú pháp JSON: {e}")
            return

    final_results = {}
    stats = {"wrapped": 0, "split_pair": 0, "split_single": 0}

    # 2. Duyệt qua từng item
    for img_id, content in data.items():
        # Lấy nội dung QA thực tế (giữ lại meta cũ nếu có, hoặc lấy trực tiếp nếu là list)
        existing_meta = {}
        if isinstance(content, dict):
            actual_qa_content = content.get("qa", [])
            # Giữ lại metadata cũ để không làm mất thông tin article_id, url
            existing_meta = {k: v for k, v in content.items() if k != "qa"}
        else:
            actual_qa_content = content

        # --- LOGIC FIX DỮ LIỆU QA ---
        # Fix trường hợp list bị bọc thêm 1 lớp: ["q", "a"] -> [["q", "a"]]
        if isinstance(actual_qa_content, list) and len(actual_qa_content) > 0:
            if isinstance(actual_qa_content[0], str):
                actual_qa_content = [actual_qa_content]
                stats["wrapped"] += 1

        new_qa_list = []
        if isinstance(actual_qa_content, list):
            for sub_list in actual_qa_content:
                if not isinstance(sub_list, list):
                    new_qa_list.append([str(sub_list).strip()])
                    continue

                # Xử lý trường hợp nhiều cặp bị dồn vào 1 list: ["q1", "a1", "q2", "a2"]
                if len(sub_list) > 2:
                    temp_pairs = []
                    is_valid = True
                    for i in range(0, len(sub_list), 2):
                        pair = sub_list[i : i+2]
                        # Kiểm tra xem có đúng là cặp Q&A không (câu đầu kết thúc bằng ?)
                        if len(pair) < 2 or not str(pair[0]).strip().endswith('?'):
                            is_valid = False
                            break
                        temp_pairs.append([str(pair[0]).strip(), str(pair[1]).strip()])
                    
                    if is_valid:
                        new_qa_list.extend(temp_pairs)
                        stats["split_pair"] += 1
                    else:
                        for item in sub_list:
                            val = str(item).strip()
                            if val: new_qa_list.append([val])
                        stats["split_single"] += 1
                else:
                    # Clean các khoảng trắng thừa
                    clean_pair = [str(x).strip() for x in sub_list if str(x).strip()]
                    if clean_pair:
                        new_qa_list.append(clean_pair)

        # --- ĐÓNG GÓI LẠI ---
        # Nếu data cũ là dict, ta cập nhật key "qa", nếu không thì lưu kết quả mới
        if existing_meta:
            final_results[img_id] = existing_meta
            final_results[img_id]["qa"] = new_qa_list
        else:
            final_results[img_id] = new_qa_list

    # 3. Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"--- BÁO CÁO KẾT QUẢ ---")
    print(f"1. Tổng số ảnh xử lý: {len(final_results)}")
    print(f"2. Fix lỗi wrapped: {stats['wrapped']}")
    print(f"3. Tách cặp Q-A (split pairs): {stats['split_pair']}")
    print(f"--- Đã lưu file tại: {output_file} ---")

if __name__ == "__main__":
    # Cấu hình file (Chỉnh sửa tên file tại đây)
    FILE_INPUT = './image_vqa.json'
    FILE_OUTPUT = './image_vqa_fixed.json' # Nên để tên khác để kiểm tra trước khi ghi đè

    fix_vqa_only(FILE_INPUT, FILE_OUTPUT)