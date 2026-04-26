import json
import os

def clean_qa_json(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Lỗi định dạng JSON: {e}")
            return

    target_phrase_answer = "Không thể"
    target_phrase_question = "Câu hỏi"
    cleaned_data = {}
    removed_count = 0

    for img_id, qa_list in data.items():
        filtered_pairs = []
        for pair in qa_list:
            # Kiểm tra cấu trúc cặp QA (phải có ít nhất 2 phần tử: Q và A)
            if not isinstance(pair, list) or len(pair) < 2:
                removed_count += 1
                continue

            question = pair[0]
            answer = pair[1]

            # Xử lý nếu Q hoặc A là list (lấy phần tử đầu tiên)
            if isinstance(question, list): question = question[0] if question else ""
            if isinstance(answer, list): answer = answer[0] if answer else ""

            # Đảm bảo dữ liệu là chuỗi để strip và so sánh
            question_text = str(question).strip()
            answer_text = str(answer).strip()

            # LOGIC LỌC: 
            # 1. Câu hỏi không được bắt đầu bằng "Câu hỏi"
            # 2. Câu trả lời không được bắt đầu bằng "Không thể"
            if question_text.startswith(target_phrase_question) or \
               answer_text.startswith(target_phrase_answer):
                removed_count += 1
            else:
                filtered_pairs.append(pair)
        
        if filtered_pairs:
            cleaned_data[img_id] = filtered_pairs

    # Ghi kết quả
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"--- Hoàn tất ---")
    print(f"Đã xóa: {removed_count} cặp không hợp lệ.")
    print(f"Dữ liệu sạch lưu tại: {output_file}")

FILE_GOC = "image_vqa_with_difficulty_cleaned.json"
FILE_MOI = "image_vqa_with_difficulty_cleaned.json" # Khuyên dùng tên khác để an toàn

if __name__ == "__main__":
    clean_qa_json(FILE_GOC, FILE_MOI)