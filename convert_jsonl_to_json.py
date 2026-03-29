import json
import os

json_path = "image_caption.json"
jsonl_path = "image_caption.jsonl"
output_path = "image_caption_updated.json"

# 1. Load file JSON gốc để lấy danh sách image_id đã có
existing_data = {}
if os.path.exists(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        print(f"[*] Đã nạp {len(existing_data)} image_id từ file JSON gốc.")
    except Exception as e:
        print(f"[!] Lỗi khi đọc file JSON: {e}")
else:
    print("[!] File JSON gốc không tồn tại. Sẽ tạo mới hoàn toàn.")

existing_ids = set(existing_data.keys())

# 2. Quét file JSONL và chỉ lấy những gì chưa có trong JSON
new_count = 0
error_count = 0

print(f"[*] Đang quét file backup {jsonl_path}...")

if os.path.exists(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                line_data = json.loads(line)
                # line_data có dạng: {"image_id": { ... }}
                for img_id, content in line_data.items():
                    if img_id not in existing_ids:
                        existing_data[img_id] = content
                        existing_ids.add(img_id)
                        new_count += 1
            except json.JSONDecodeError:
                # Bỏ qua dòng bị lỗi/ghi dở do kill -9
                error_count += 1
else:
    print(f"[!] File backup {jsonl_path} không tìm thấy.")

# 3. Lưu lại file JSON duy nhất đã được cập nhật
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, indent=2, ensure_ascii=False)

print("-" * 30)
print(f"[OK] Hoàn tất cập nhật!")
print(f"- Số lượng cũ: {len(existing_data) - new_count}")
print(f"- Số lượng mới nạp thêm từ JSONL: {new_count}")
print(f"- Dòng JSONL bị lỗi bỏ qua: {error_count}")
print(f"- Tổng cộng hiện có: {len(existing_data)} entries.")
print(f"- File mới: {output_path}")