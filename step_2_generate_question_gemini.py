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
        generated_caption=entry.get("generated_caption", "")[:2000],
    )

def save_json(data, path):
    temp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)

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
        
        valid_extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".webp"]
        image_path = None
        for ext in valid_extensions:
            test_path = os.path.join(args.image_dir, f"{image_id}{ext}")
            if os.path.exists(test_path):
                image_path = test_path
                break
            
        total_len = (
            len(str(entry.get("title", ""))) + 
            len(str(entry.get("original_caption", ""))) + 
            len(str(entry.get("generated_caption", "")))
        )

        if os.path.exists(image_path):
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "entry": entry,
                "total_len": total_len
            })
            
    all_tasks.sort(key=lambda x: x["total_len"])
    
    # all_tasks.sort(key=lambda x: x["total_len"], reverse=True)

    if not all_tasks:
        print("[*] No new images to process.")
        return
    
    backup_path = args.output_vqa_path.replace(".json", ".jsonl")

    if args.limit > 0:
        all_tasks = all_tasks[:args.limit]
        print(f"[*] Limit applied: Only processing the first {len(all_tasks)} tasks.")

    print(f"[*] Total tasks to process: {len(all_tasks)}")

    # 5. Batch Processing (Sử dụng Multi-threading API)
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Gemini VQA Processing")

    try:
        # Mở file backup chế độ 'append' (a)
        with open(backup_path, "a", encoding="utf-8") as f_backup:
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i : i + batch_size]
                
                batch_prompts = [render_vqa_prompt(args.template_dir, t['entry']) for t in batch]
                batch_img_paths = [t['image_path'] for t in batch]
                batch_ids = [t['image_id'] for t in batch]
                
                for idx, p in enumerate(batch_prompts):
                    print(f"[*] Image ID: {batch[idx]['image_id']} | Prompt Length: {len(p)} characters")

                try:
                    # Gọi API song song
                    batch_results = model.generate_questions_batch(
                        batch_img_paths, 
                        prompts=batch_prompts,
                        max_workers=args.max_workers
                    )

                    for idx, questions in enumerate(batch_results):
                        img_id = batch_ids[idx]
                        if questions and isinstance(questions, list):
                            vqa_list = questions
                            
                            # Lưu vào memory
                            vqa_output[img_id] = vqa_list
                            updated_count += 1
                            
                            # Ghi ngay vào file backup .jsonl
                            line = json.dumps({img_id: vqa_list}, ensure_ascii=False)
                            f_backup.write(line + "\n")
                    
                    # Quan trọng: Đẩy dữ liệu từ RAM xuống đĩa cứng
                    f_backup.flush()

                    # Lưu file .json chính định kỳ (mỗi 10 batch) để tránh mất mát lớn
                    if (i // batch_size) % 10 == 0:
                        save_json(vqa_output, args.output_vqa_path)

                except Exception as e:
                    print(f"\n[!] Error at batch starting with {batch_ids[0]}: {e}")
                    # Tùy chọn: time.sleep(5) nếu bị rate limit API
                
                pbar.update(len(batch))

    except KeyboardInterrupt:
        print("\n[!] User Interrupted (^C). Saving progress...")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
    finally:
        # 5. Final Save: Luôn chạy dù script crash hay bị dừng manual
        if updated_count > 0:
            save_json(vqa_output, args.output_vqa_path)
            print(f"[*] Final save completed. Total entries: {len(vqa_output)}")
        pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_json_path", type=str, default="./image_caption_updated.json")
    parser.add_argument("--output_vqa_path", type=str, default="./image_questions.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-lite")
    
    # Limit
    parser.add_argument("--limit", type=int, default=10000, help="Giới hạn số lượng ảnh xử lý mới")
    
    # Threading options (API chịu được khoảng 5-10 request song song tùy loại tài khoản)
    parser.add_argument("--batch_size", type=int, default=4, help="Số lượng task xử lý trước khi lưu file")
    parser.add_argument("--max_workers", type=int, default=4, help="Số lượng request API gửi đi song song")
    
    args = parser.parse_args()
    process_vqa_pipeline(args)

if __name__ == "__main__":
    main()