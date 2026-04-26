import json
import os
import argparse

def sync_captions(database_path, caption_path, output_path):
    # 1. Load merged_2_database.json
    if not os.path.exists(database_path):
        print(f"[Error] Không tìm thấy file database: {database_path}")
        return

    print(f"[*] Đang đọc database từ: {database_path}")
    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    # 2. Thu thập tất cả các image_id hợp lệ từ database mới
    valid_image_ids = set()
    for art_id, art in database.items():
        for img in art.get("images", []):
            image_id = img.get("image_id")
            if image_id:
                valid_image_ids.add(image_id)
    
    print(f"[*] Tìm thấy {len(valid_image_ids)} image_id hợp lệ trong database.")

    # 3. Load image_caption_updated.json
    if not os.path.exists(caption_path):
        print(f"[Error] Không tìm thấy file caption: {caption_path}")
        return

    with open(caption_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    
    initial_count = len(caption_data)
    print(f"[*] Đang kiểm tra {initial_count} entry trong file caption hiện tại...")

    # 4. Lọc bỏ các entry không còn nằm trong valid_image_ids
    # Tạo dictionary mới chỉ chứa các key có trong database
    updated_caption_data = {
        img_id: data for img_id, data in caption_data.items() 
        if img_id in valid_image_ids
    }

    final_count = len(updated_caption_data)
    removed_count = initial_count - final_count
    
    missing_ids = valid_image_ids - set(updated_caption_data.keys())
    print(f"Danh sách 36 ID chưa có caption: {list(missing_ids)}...")

    # 5. Lưu lại kết quả
    if removed_count > 0:
        print(f"[!] Đã loại bỏ {removed_count} entry không còn tồn tại trong database.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(updated_caption_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] Đã cập nhật file thành công: {output_path}")
        print(f"[*] Số lượng entry còn lại: {final_count}")
    else:
        print("[*] Không có entry nào cần xóa. File caption đã đồng bộ với database.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đồng bộ file caption với database sau khi xóa image_id.")
    parser.add_argument("--database", type=str, default="../Eventa/webCrawl/src/merged_7_database.json", help="Đường dẫn file database gốc")
    parser.add_argument("--caption", type=str, default="./image_caption_updated.json", help="Đường dẫn file caption cần cập nhật")
    parser.add_argument("--output", type=str, default="./image_caption_updated.json", help="Nơi lưu file sau khi cập nhật (mặc định ghi đè)")

    args = parser.parse_args()
    sync_captions(args.database, args.caption, args.output)