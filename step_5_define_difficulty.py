import os
import json
import argparse
import time
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel 

def render_difficulty_prompt(template_dir, article_content, qa_pairs):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("difficulty_rating.j2")
    
    # Truncate content to avoid context overflow
    truncated_content = article_content[:5000] 
    
    return template.render(
        content=truncated_content,
        qa_pairs=qa_pairs
    )

def save_json(data, path):
    """Lưu file JSON an toàn"""
    temp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)

def find_image_path(image_dir, image_id):
    """Tìm đường dẫn ảnh"""
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.webp']:
        path = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def process_difficulty_pipeline(args):
    # 1. Load Data
    print("[*] Loading data files for difficulty rating...")
    if not os.path.exists(args.vqa_json_path):
        print(f"[Error] VQA input file not found: {args.vqa_json_path}")
        return

    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Resume logic
    all_tasks = []
    for image_id, qa_list in vqa_data.items():
        # Kiểm tra nếu đã có difficulty (đã có element thứ 3 trong list [Q, A, D])
        if len(qa_list) > 0 and len(qa_list[0]) >= 3:
            continue

        entry = caption_data.get(image_id)
        image_path = find_image_path(args.image_dir, image_id)
        
        if entry and image_path:
            article_id = entry.get("article_id", image_id) 
            article_info = database.get(article_id, {})
            article_content = article_info.get("content", "Không có nội dung.")
            
            prompt = render_difficulty_prompt(args.template_dir, article_content, qa_list)
            
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "prompt": prompt,
                "prompt_len": len(prompt),
                "qa_list": qa_list
            })

    if not all_tasks:
        print("[*] All QA pairs already have difficulty ratings.")
        return
    
    # Sắp xếp theo độ dài prompt để tối ưu hóa GPU
    all_tasks.sort(key=lambda x: x['prompt_len'], reverse=args.reverse_sort)
    
    if args.limit and args.limit > 0:
        all_tasks = all_tasks[:args.limit]
        print(f"[*] Limit applied: Processing {len(all_tasks)} tasks.")

    # Tạo đường dẫn file backup .jsonl
    backup_path = args.output_json_path.replace(".json", ".jsonl")
    print(f"[*] Total images to process: {len(all_tasks)}")
    print(f"[*] Backup file: {backup_path}")

    # 3. Khởi tạo Model Local
    print(f"[*] Initializing model: {args.model_name} on {args.device}")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 4. Batch Processing
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Local Difficulty Rating")

    try:
        # Mở file backup .jsonl ở chế độ append
        with open(backup_path, "a", encoding="utf-8") as f_backup:
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i : i + batch_size]
                batch_prompts = [t['prompt'] for t in batch]
                batch_img_paths = [t['image_path'] for t in batch]
                batch_ids = [t['image_id'] for t in batch]
                
                try:
                    # Inference
                    batch_results = model.generate_difficulty_batch(
                        image_paths=batch_img_paths, 
                        prompts=batch_prompts
                    )

                    for idx, result in enumerate(batch_results):
                        img_id = batch_ids[idx]
                        original_qa = batch[idx]["qa_list"]
                        
                        if result and isinstance(result, list) and len(result) == len(original_qa):
                            updated_qa = []
                            for qa_idx, qa_pair in enumerate(original_qa):
                                new_pair = list(qa_pair[:2])
                                new_pair.append(str(result[qa_idx]))
                                updated_qa.append(new_pair)
                            
                            vqa_data[img_id] = updated_qa
                            updated_count += 1
                            
                            # Ghi backup ngay lập tức
                            f_backup.write(json.dumps({img_id: updated_qa}, ensure_ascii=False) + "\n")
                        else:
                            print(f"\n[!] Warning: Result mismatch for image {img_id}. Skipping.")

                    # Flush dữ liệu xuống đĩa và lưu file chính định kỳ
                    f_backup.flush()
                    if (i // batch_size) % 5 == 0:
                        save_json(vqa_data, args.output_json_path)
                        if "cuda" in args.device:
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[!] Error at batch starting with {batch_ids[0]}: {e}")
                
                pbar.update(len(batch))
                
    except KeyboardInterrupt:
        print("\n[!] User Interrupted. Saving progress...")
    finally:
        # 5. Final Save
        if updated_count > 0:
            save_json(vqa_data, args.output_json_path)
            print(f"[*] Done! Processed {updated_count} entries. Total: {len(vqa_data)}")
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption_updated.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_vqa.json")
    parser.add_argument("--output_json_path", type=str, default="./image_vqa_with_difficulty.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    
    # Model config
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reverse_sort", action="store_true", help="Sort prompts by length descending")

    args = parser.parse_args()

    process_difficulty_pipeline(args)
