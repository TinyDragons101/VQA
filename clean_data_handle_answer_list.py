import json
import os

def process_qa_file(input_path, output_path):
    # 1. Kiểm tra file đầu vào có tồn tại không
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file {input_path}")
        return

    try:
        # 2. Đọc dữ liệu từ file JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        cleaned_data = {}

        # 3. Duyệt qua từng key (ID) và danh sách các cặp QA
        for entry_id, qa_list in data.items():
            new_qa_list = []
            
            for pair in qa_list:
                # pair[0] là câu hỏi, pair[1] là câu trả lời
                question = pair[0]
                answer = pair[1]

                # Kiểm tra nếu answer là một list thì nối lại thành chuỗi
                if isinstance(answer, list):
                    # Nối bằng dấu phẩy và khoảng trắng
                    answer = ", ".join(map(str, answer))
                
                new_qa_list.append([question, answer])
            
            cleaned_data[entry_id] = new_qa_list

        # 4. Ghi dữ liệu đã làm sạch ra file output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        
        print(f"Thành công! Đã lưu dữ liệu sạch tại: {output_path}")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    # Cấu hình tên file input và output tại đây
    INPUT_FILE = "./image_vqa.json"
    OUTPUT_FILE = "./image_vqa.json"
    
    process_qa_file(INPUT_FILE, OUTPUT_FILE)