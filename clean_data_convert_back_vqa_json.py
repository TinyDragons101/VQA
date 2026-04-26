import json

# Đọc file dữ liệu gốc
file_name = 'image_vqa.json'
with open(file_name, 'r', encoding='utf-8') as f:
    old_data = json.load(f)

new_data = {}
total_images = 0
total_qa_pairs = 0

for img_id, content in old_data.items():
    if "qa" in content:
        # Gán danh sách QA vào image_id
        new_data[img_id] = content["qa"]
        
        # Cập nhật thống kê
        total_images += 1
        total_qa_pairs += len(content["qa"])

# Lưu file kết quả
with open('image_vqa.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

# Xuất thông tin thống kê
print(f"--- BÁO CÁO CHUYỂN ĐỔI ---")
print(f"Tổng số image_id đã xử lý: {total_images}")
print(f"Tổng số cặp câu hỏi - trả lời: {total_qa_pairs}")
print(f"Trung bình: {total_qa_pairs/total_images:.2f} câu hỏi/ảnh")
print(f"File đã được lưu tại: image_vqa.json")