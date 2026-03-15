import json
import argparse
import os
import re
import time
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from dotenv import load_dotenv
from gemini import GeminiVLCaptionModel

# Load môi trường (API Key)
load_dotenv()

def render_vqa_prompt(template_dir, entry):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("question_generating_gemini.j2")
    return template.render(
        title=entry.get("title", ""),
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", ""),
    )

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_vqa_pipeline(args):
    # 1. Load data
    if not os.path.exists(args.caption_json_path):
        print(f"[Error] Phase 1 output not found: {args.caption_json_path}")
        return

    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    # 2. Resume logic
    vqa_output = {}
    if os.path.exists(args.output_vqa_path):
        try:
            with open(args.output_vqa_path, "r", encoding="utf-8") as f:
                vqa_output = json.load(f)
            print(f"[*] Resuming: {len(vqa_output)} images already processed.")
        except:
            vqa_output = {}

    # 3. Init Gemini Model
    model = GeminiVLCaptionModel(model_name=args.model_name)

    # 4. Filter tasks
    all_tasks = []
    for image_id, entry in caption_data.items():
        if image_id in vqa_output:
            continue
        
        # Hỗ trợ cả .png và .jpg tùy dataset của bạn
        image_path = os.path.join(args.image_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(args.image_dir, f"{image_id}.jpg")

        if os.path.exists(image_path):
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "entry": entry
            })

    if not all_tasks:
        print("[*] No new images to process.")
        return

    print(f"[*] Total tasks to process: {len(all_tasks)}")

    # 5. Batch Processing (Sử dụng Multi-threading API)
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Gemini VQA Processing")

    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        
        batch_prompts = [render_vqa_prompt(args.template_dir, t['entry']) for t in batch]
        batch_img_paths = [t['image_path'] for t in batch]
        batch_ids = [t['image_id'] for t in batch]

        try:
            # Gọi API song song cho nguyên một batch
            batch_results = model.generate_questions_batch(
                batch_img_paths, 
                prompts=batch_prompts,
                max_workers=args.max_workers
            )

            for idx, questions in enumerate(batch_results):
                img_id = batch_ids[idx]
                if questions and isinstance(questions, list):
                    vqa_list = [{"question": q, "answer": ""} for q in questions]
                    vqa_output[img_id] = vqa_list
                    updated_count += 1

            # Lưu định kỳ
            if updated_count > 0 and (i // batch_size) % 5 == 0:
                save_json(vqa_output, args.output_vqa_path)

        except Exception as e:
            print(f"\n[!] Critical error at batch {i}: {e}")
        
        pbar.update(len(batch))

    pbar.close()

    # 6. Final Save
    save_json(vqa_output, args.output_vqa_path)
    print(f"\n[Summary] Total VQA entries: {len(vqa_output)} | Newly Generated: {updated_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_json_path", type=str, default="./image_caption3.json")
    parser.add_argument("--output_vqa_path", type=str, default="./image_vqa3.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-lite")
    
    # Threading options (API chịu được khoảng 5-10 request song song tùy loại tài khoản)
    parser.add_argument("--batch_size", type=int, default=1, help="Số lượng task xử lý trước khi lưu file")
    parser.add_argument("--max_workers", type=int, default=1, help="Số lượng request API gửi đi song song")
    
    args = parser.parse_args()
    process_vqa_pipeline(args)

if __name__ == "__main__":
    main()