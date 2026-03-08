import json
import argparse
import os
import torch
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel

def render_answer_prompt(template_dir, entry, article_content, questions):
    """
    Render prompt với danh sách câu hỏi sử dụng Jinja2.
    """
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("answer_generating.j2")
    return template.render(
        title=entry.get("title", ""),
        content=article_content[:3000],
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", ""),
        questions=questions
    )

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_image_path(image_dir, image_id):
    """
    Hỗ trợ tìm file ảnh với nhiều định dạng phổ biến.
    """
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']:
        path = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def process_answering_pipeline(args):
    # 1. Load dữ liệu
    print("[*] Loading data files...")
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    if not os.path.exists(args.vqa_json_path):
        print(f"[Error] VQA questions file not found: {args.vqa_json_path}")
        return
        
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Khởi tạo Model
    print(f"[*] Initializing model: {args.model_name}")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    updated_images_count = 0
    
    # 3. Duyệt qua từng hình ảnh trong dataset VQA
    for image_id, vqa_list in tqdm(vqa_data.items(), desc="Answering Questions"):
        
        # Kiểm tra những câu hỏi chưa có câu trả lời (answer rỗng)
        unanswered_indices = [i for i, item in enumerate(vqa_list) if not item.get("answer")]
        if not unanswered_indices:
            continue

        # Lấy metadata của ảnh
        entry = caption_data.get(image_id)
        if not entry:
            print(f"\n[!] Missing metadata for image {image_id}")
            continue
        
        # Tìm đường dẫn ảnh
        image_path = find_image_path(args.image_dir, image_id)
        if not image_path:
            print(f"\n[!] Image file not found for {image_id}")
            continue

        # Lấy nội dung bài báo từ database
        article_id = entry.get("article_id")
        article_content = database.get(article_id, {}).get("content", "Không có nội dung bài báo.")
        
        # Danh sách các câu hỏi cần trả lời cho ảnh này
        questions_to_ask = [vqa_list[i]["question"] for i in unanswered_indices]

        try:
            # Render prompt duy nhất cho tất cả câu hỏi của ảnh này
            prompt = render_answer_prompt(
                args.template_dir, 
                entry, 
                article_content, 
                questions_to_ask
            )
            
            # Gọi model 1 lần để sinh toàn bộ câu trả lời dưới dạng list (JSON)
            answers = model.generate_answers(image_path, prompt)
            
            if answers and isinstance(answers, list) and len(answers) == len(questions_to_ask):
                # Map câu trả lời vào đúng vị trí trong vqa_list
                for idx, ans in zip(unanswered_indices, answers):
                    vqa_list[idx]["answer"] = str(ans).strip()
                
                updated_images_count += 1
            else:
                print(f"\n[!] Warning: Output mismatch or format error for {image_id}. "
                      f"Expected {len(questions_to_ask)} answers, got {len(answers) if answers else 0}.")

            # Lưu checkpoint định kỳ và dọn dẹp cache
            if updated_images_count % 10 == 0:
                save_json(vqa_data, args.vqa_json_path)
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] Error at {image_id}: {e}")

    # 4. Lưu kết quả cuối cùng
    save_json(vqa_data, args.vqa_json_path)
    print(f"\n[+] Hoàn thành! Đã cập nhật câu trả lời cho {updated_images_count} hình ảnh.")

def main():
    parser = argparse.ArgumentParser(description="VQA Answering Pipeline - Batch processing per image")
    parser.add_argument("--database_path", type=str, default="./database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="./images")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    process_answering_pipeline(args)

if __name__ == "__main__":
    main()