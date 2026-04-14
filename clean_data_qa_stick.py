import json
import os

def fix_vqa_with_metadata(input_file, output_file, caption_file, db_file):
    print(f"--- Đang khởi động xử lý dữ liệu: {input_file} ---")
    
    if not all(os.path.exists(f) for f in [input_file, caption_file, db_file]):
        print("Lỗi: Một trong các file đầu vào không tồn tại.")
        return

    # 1. Load Mapping Data
    print("Loading metadata mapping...")
    with open(caption_file, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
    with open(db_file, 'r', encoding='utf-8') as f:
        db_data = json.load(f)

    # Tạo dictionary tra cứu nhanh: image_id -> {article_id, article_url}
    img_to_meta = {}
    for img_id, info in caption_data.items():
        art_id = info.get("article_id")
        url = db_data.get(art_id, {}).get("url", "")
        img_to_meta[img_id] = {"article_id": art_id, "article_url": url}

    # 2. Đọc file VQA cần fix
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Lỗi cú pháp JSON: {e}")
            return

    final_results = {}
    stats = {"wrapped": 0, "split_pair": 0, "split_single": 0}

    # Duyệt qua từng item trong file VQA
    for img_id, content in data.items():
        # Nếu content đang ở format cũ (có sẵn article_id), ta lấy list qa ra xử lý
        if isinstance(content, dict) and "qa" in content:
            actual_qa_content = content["qa"]
        else:
            actual_qa_content = content

        # --- LOGIC FIX DỮ LIỆU CỦA BẠN ---
        if isinstance(actual_qa_content, list) and len(actual_qa_content) > 0:
            if isinstance(actual_qa_content[0], str):
                actual_qa_content = [actual_qa_content]
                stats["wrapped"] += 1

        new_qa_list = []
        for sub_list in actual_qa_content:
            if not isinstance(sub_list, list):
                new_qa_list.append([str(sub_list).strip()])
                continue

            if len(sub_list) > 2:
                temp_pairs = []
                is_valid = True
                for i in range(0, len(sub_list), 2):
                    pair = sub_list[i : i+2]
                    if not str(pair[0]).strip().endswith('?') or len(pair) < 2:
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
                clean_pair = [str(x).strip() for x in sub_list if str(x).strip()]
                if clean_pair:
                    new_qa_list.append(clean_pair)

        # --- BƯỚC 3: ĐÓNG GÓI THEO FORMAT MỚI ---
        meta = img_to_meta.get(img_id, {"article_id": None, "article_url": None})
        final_results[img_id] = {
            "article_id": meta["article_id"],
            "article_url": meta["article_url"],
            "qa": new_qa_list
        }

    # 4. Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"--- BÁO CÁO KẾT QUẢ ---")
    print(f"1. Tổng số ảnh xử lý: {len(final_results)}")
    print(f"2. Fix lỗi wrapped: {stats['wrapped']} | Split pairs: {stats['split_pair']}")
    print(f"--- Đã lưu file tại: {output_file} ---")

if __name__ == "__main__":
    # Cấu hình file
    FILE_INPUT = './image_vqa_folder/image_vqa_01.json'
    FILE_OUTPUT = './image_vqa_folder/image_vqa_01_fixed.json'
    
    # Cần thêm 2 file này để lấy metadata
    FILE_CAPTION = 'image_caption_updated.json'
    FILE_DB = "../Eventa/webCrawl/src/merged_2_database.json"

    fix_vqa_with_metadata(FILE_INPUT, FILE_OUTPUT, FILE_CAPTION, FILE_DB)