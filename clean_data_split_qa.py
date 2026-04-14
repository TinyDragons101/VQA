import json
import os

def split_and_combine_vqa(vqa_file, caption_file, db_file, output_folder, chunk_size=2500):
    # 1. Tạo folder đầu ra
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Đọc các file bổ trợ để lấy mapping
    print("Loading mapping data...")
    
    # Load image_caption để biết image_id thuộc article_id nào
    with open(caption_file, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
    
    # Load database để lấy URL của article_id
    with open(db_file, 'r', encoding='utf-8') as f:
        db_data = json.load(f)

    # Tạo dictionary tra cứu nhanh: image_id -> {article_id, url}
    img_to_article = {}
    for img_id, info in caption_data.items():
        art_id = info.get("article_id")
        url = db_data.get(art_id, {}).get("url", "")
        img_to_article[img_id] = {
            "article_id": art_id,
            "article_url": url
        }

    # 3. Đọc và xử lý file VQA chính
    print(f"Processing {vqa_file}...")
    with open(vqa_file, 'r', encoding='utf-8') as f:
        vqa_data = json.load(f)

    # Sắp xếp theo số lượng QA (tăng dần/giảm dần tùy bạn, ở đây giữ nguyên logic cũ)
    sorted_items = sorted(vqa_data.items(), key=lambda x: len(x[1]))
    total_items = len(sorted_items)

    # 4. Chia nhỏ và lưu file với cấu trúc mới
    file_count = 1
    for i in range(0, total_items, chunk_size):
        chunk_items = sorted_items[i : i + chunk_size]
        new_chunk = {}

        for img_id, qa_list in chunk_items:
            # Lấy thông tin article từ mapping đã tạo ở bước 2
            meta = img_to_article.get(img_id, {"article_id": None, "article_url": None})
            
            new_chunk[img_id] = {
                "article_id": meta["article_id"],
                "article_url": meta["article_url"],
                "qa": qa_list
            }

        # Lưu file
        file_name = f"image_vqa_{file_count:02d}.json"
        file_path = os.path.join(output_folder, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_chunk, f, indent=4, ensure_ascii=False)
        
        print(f"Saved: {file_name} ({len(new_chunk)} images)")
        file_count += 1

# --- Chạy script ---
if __name__ == "__main__":
    # Đường dẫn các file đầu vào của bạn
    INPUT_VQA = 'image_vqa.json'
    INPUT_CAPTION = 'image_caption_updated.json'
    INPUT_DB = "../Eventa/webCrawl/src/merged_2_database.json"
    OUTPUT_FOLDER = 'image_vqa_folder'
    
    split_and_combine_vqa(INPUT_VQA, INPUT_CAPTION, INPUT_DB, OUTPUT_FOLDER)