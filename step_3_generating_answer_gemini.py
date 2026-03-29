import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
from gemini import GeminiVLCaptionModel

# Load môi trường (API Key)
load_dotenv()

def render_answer_prompt(template_dir, entry, article_content, questions):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("answer_generating_gemini.j2")
    
    # Giới hạn nội dung bài báo để tránh tràn token nếu bài quá dài
    truncated_content = article_content[:5000] 
    
    return template.render(
        title=entry.get("title", ""),
        content=truncated_content,
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", ""),
        questions=questions
    )

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_image_path(image_dir, image_id):
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']:
        path = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def process_answering_pipeline(args):
    # 1. Load Data
    print("[*] Loading data files...")
    if not os.path.exists(args.vqa_json_path):
        print(f"[Error] VQA input file not found: {args.vqa_json_path}")
        return

    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # Load output cũ để resume nếu cần (hoặc tạo mới)
    output_data = {}
    if os.path.exists(args.output_json_path):
        with open(args.output_json_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

    # 2. Chuẩn bị Task
    all_tasks = []
    for image_id, questions in vqa_data.items():
        # Bỏ qua nếu đã xử lý rồi (Resume logic)
        if image_id in output_data and output_data[image_id]:
            continue

        entry = caption_data.get(image_id)
        image_path = find_image_path(args.image_dir, image_id)
        
        if entry and image_path:
            # Lấy nội dung bài báo từ database qua article_id (hoặc id trực tiếp)
            article_id = entry.get("article_id", image_id) 
            article_info = database.get(article_id, {})
            article_content = article_info.get("content", "Không có nội dung.")
            
            prompt = render_answer_prompt(args.template_dir, entry, article_content, questions)
            
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "prompt": prompt
            })

    if not all_tasks:
        print("[*] No new images/questions to process.")
        return

    print(f"[*] Total images to process: {len(all_tasks)}")

    # 3. Khởi tạo Model
    model = GeminiVLCaptionModel(model_name=args.model_name)

    # 4. Batch Processing
    batch_size = args.batch_size
    pbar = tqdm(total=len(all_tasks), desc="Gemini Answering")

    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        
        batch_prompt = [t['prompt'] for t in batch]
        batch_img_paths = [t['image_path'] for t in batch]
        batch_ids = [t['image_id'] for t in batch]

        try:
            # Lưu ý: Hàm generate_answers_batch cần trả về list các list [Q, A]
            # Bạn có thể cần parse JSON từ chuỗi text mà Gemini trả về trong model.py
            batch_results = model.generate_answers_batch(
                tasks=batch, 
                max_workers=args.workers
            )

            for idx, result in enumerate(batch_results):
                img_id = batch_ids[idx]
                if result and isinstance(result, list):
                    # Lưu lại format [["Q","A"], ["Q","A"]]
                    output_data[img_id] = result

            # Lưu định kỳ
            if (i // batch_size) % 5 == 0:
                save_json(output_data, args.output_json_path)

        except Exception as e:
            print(f"\n[!] Error at batch starting with {batch_ids[0]}: {e}")
        
        pbar.update(len(batch))

    pbar.close()

    # 5. Final Save
    save_json(output_data, args.output_json_path)
    print(f"\n[DONE] Processed {len(output_data)} images. Output saved to {args.output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption3.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_question3.json")
    parser.add_argument("--output_json_path", type=str, default="./image_vqa3.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-lite")
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Số luồng API chạy song song")
    
    args = parser.parse_args()
    process_answering_pipeline(args)