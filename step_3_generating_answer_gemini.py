import os
import json
import time
import argparse
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from dotenv import load_dotenv
from gemini import GeminiVLCaptionModel

load_dotenv()

def render_answer_prompt(template_dir, entry, article_content, questions):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    # File template này cần nhận biến 'questions' là một list
    template = env.get_template("answer_generating_gemini.j2")
    return template.render(
        title=entry.get("title", ""),
        content=article_content[:3500], # Gemini hỗ trợ context lớn hơn nên có thể tăng lên
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
        if os.path.exists(path): return path
    return None

def process_answering_pipeline(args):
    # 1. Load Data
    print("[*] Loading data files...")
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Chuẩn bị Task
    all_tasks = []
    for image_id, vqa_list in vqa_data.items():
        unanswered_indices = [i for i, item in enumerate(vqa_list) if not item.get("answer")]
        if not unanswered_indices: continue

        entry = caption_data.get(image_id)
        image_path = find_image_path(args.image_dir, image_id)
        
        if entry and image_path:
            article_id = entry.get("article_id")
            article_content = database.get(article_id, {}).get("content", "Không có nội dung.")
            questions_to_ask = [vqa_list[i]["question"] for i in unanswered_indices]
            
            prompt = render_answer_prompt(args.template_dir, entry, article_content, questions_to_ask)
            
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "prompt": prompt,
                "unanswered_indices": unanswered_indices
            })

    if not all_tasks:
        print("[*] No questions to answer.")
        return

    # 3. Khởi tạo Model
    model = GeminiVLCaptionModel(model_name=args.model_name)

    # 4. Chạy theo từng nhóm nhỏ (Mini-batches) để tránh mất dữ liệu khi lỗi giữa chừng
    chunk_size = args.save_every 
    for i in range(0, len(all_tasks), chunk_size):
        current_batch_tasks = all_tasks[i : i + chunk_size]
        
        # Gọi API song song
        batch_results = model.generate_answers_parallel(current_batch_tasks, max_workers=args.workers)

        # 5. Cập nhật kết quả
        updated_in_chunk = 0
        for task, answers in zip(current_batch_tasks, batch_results):
            if answers and len(answers) == len(task['unanswered_indices']):
                img_id = task['image_id']
                for q_idx, ans_text in zip(task['unanswered_indices'], answers):
                    vqa_data[img_id][q_idx]["answer"] = str(ans_text).strip()
                updated_in_chunk += 1
        
        # Lưu sau mỗi chunk
        save_json(vqa_data, args.vqa_json_path)
        print(f"[✔] Progress: {i + len(current_batch_tasks)}/{len(all_tasks)} - Saved {updated_in_chunk} items.")

    print(f"\n[DONE] Pipeline done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_captio3.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_vqa3.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--workers", type=int, default=1, help="Số luồng API chạy song song")
    parser.add_argument("--save_every", type=int, default=20, help="Lưu lại file sau mỗi X ảnh")
    
    args = parser.parse_args()
    process_answering_pipeline(args)