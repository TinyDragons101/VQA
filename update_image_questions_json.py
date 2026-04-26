import json
import os
import argparse

def sync_questions(database_path, questions_path, output_path):
    # 1. Load merged_database (Nguồn xác thực duy nhất)
    if not os.path.exists(database_path):
        print(f"[Error] Không tìm thấy file database: {database_path}")
        return

    print(f"[*] Đang đọc database từ: {database_path}")
    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    # 2. Thu thập tập hợp các image_id hợp lệ
    valid_image_ids = set()
    for art_id, art in database.items():
        for img in art.get("images", []):
            img_id = img.get("image_id")
            if img_id:
                valid_image_ids.add(img_id)
    
    print(f"[*] Tổng số image_id hợp lệ trong database: {len(valid_image_ids)}")

    # 3. Load file câu hỏi cần đồng bộ
    if not os.path.exists(questions_path):
        print(f"[Error] Không tìm thấy file: {questions_path}")
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    initial_count = len(questions_data)
    print(f"[*] Đang kiểm tra {initial_count} entry trong file questions...")

    # 4. Lọc: Chỉ giữ lại những image_id còn tồn tại trong database
    updated_questions = {
        img_id: content for img_id, content in questions_data.items() 
        if img_id in valid_image_ids
    }

    final_count = len(updated_questions)
    removed_count = initial_count - final_count

    # 5. Lưu kết quả
    if removed_count > 0:
        print(f"[!] Đã loại bỏ {removed_count} ID không còn tồn tại.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(updated_questions, f, indent=2, ensure_ascii=False)
        print(f"[OK] Đã cập nhật xong: {output_path}")
        print(f"[*] Số lượng còn lại: {final_count}")
    else:
        print("[*] File đã đồng bộ sẵn, không cần xóa gì thêm.")

    # Kiểm tra xem có ID nào có trong database mà chưa có trong file questions không
    missing_ids = valid_image_ids - set(updated_questions.keys())
    if missing_ids:
        print(f"[⚠️] Lưu ý: Còn {len(missing_ids)} ID trong database chưa được tạo câu hỏi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đồng bộ file questions với database.")
    parser.add_argument("--database", type=str, default="../Eventa/webCrawl/src/merged_7_database.json")
    parser.add_argument("--questions", type=str, default="./image_questions.json", help="File questions cần lọc")
    parser.add_argument("--output", type=str, default="./image_questions_updated.json")

    args = parser.parse_args()
    sync_questions(args.database, args.questions, args.output)