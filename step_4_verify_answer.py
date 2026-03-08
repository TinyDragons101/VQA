import json
import argparse
import os
import torch
import re
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel

def render_verification_prompt(template_dir, entry, article_content, qa_pairs):
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("verification_generating.j2")
    return template.render(
        title=entry.get("title", ""),
        content=article_content[:3500], # Giới hạn để tránh tràn context
        qa_pairs=qa_pairs
    )

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_verification_pipeline(args):
    # 1. Load dữ liệu
    print("[*] Loading data for verification...")
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Khởi tạo Model
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    updated_count = 0
    
    # 3. Duyệt qua từng ảnh để verify toàn bộ QA
    for image_id, vqa_list in tqdm(vqa_data.items(), desc="Verifying Ground Truth"):
        
        # Kiểm tra nếu ảnh này đã có explanation cho tất cả câu hỏi thì bỏ qua
        needs_verify = [i for i, item in enumerate(vqa_list) if "explanation" not in item]
        if not needs_verify:
            continue

        entry = caption_data.get(image_id)
        if not entry: continue
        
        article_id = entry.get("article_id")
        article_data = database.get(article_id, {})
        article_content = article_data.get("content", "")
        
        # Chỉ lấy ảnh nếu thực sự cần (mặc dù bước verify này chủ yếu dựa trên text)
        image_path = os.path.join(args.image_dir, f"{image_id}.png")
        if not os.path.exists(image_path): continue

        try:
            # Render prompt cho toàn bộ list QA của ảnh này
            prompt = render_verification_prompt(
                args.template_dir, 
                entry, 
                article_content, 
                vqa_list # Gửi toàn bộ để model audit
            )

            # Gọi model xử lý (Sử dụng hàm generate_explanations tương tự generate_answers)
            # Chúng ta dùng chung hàm extract_json_array trong class của bạn
            explanations = model.generate_answers(image_path, prompt)

            if explanations and isinstance(explanations, list) and len(explanations) == len(vqa_list):
                for i in range(len(vqa_list)):
                    vqa_list[i]["explanation"] = str(explanations[i]).strip()
                updated_count += 1
            else:
                print(f"\n[!] Verification format error for {image_id}. Expected {len(vqa_list)} items.")

            # Lưu định kỳ
            if updated_count % 10 == 0:
                save_json(vqa_data, args.vqa_json_path)
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] Error verifying {image_id}: {e}")

    # 4. Lưu kết quả cuối cùng
    save_json(vqa_data, args.vqa_json_path)
    print(f"\n[+] Hoàn thành verify cho {updated_count} hình ảnh.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="./database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="./images")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    process_verification_pipeline(args)

if __name__ == "__main__":
    main()