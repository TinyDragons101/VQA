import os
import json
import argparse
import time
import torch
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel 

def render_difficulty_prompt(template_dir, article_content, qa_pair):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("difficulty_rating.j2")
    
    # Truncate content to avoid context overflow
    truncated_content = article_content[:3000] 
    
    return template.render(
        content=truncated_content,
        question=qa_pair.get("question", ""),
        answer=qa_pair.get("answer", "")
    )

def process_difficulty_pipeline(args):
    # 1. Load Data
    print("[*] Loading VQA data to add difficulty...")
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(args.vqa_output_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Chuẩn bị danh sách Task (Flattening)
    # Vì một image_id có nhiều cặp QA, ta "phẳng hóa" để xử lý batch hiệu quả hơn
    all_qa_tasks = []
    for image_id, qa_list in vqa_data.items():
        # Kiểm tra xem ảnh có tồn tại không
        image_path = find_image_path(args.image_dir, image_id)
        entry = caption_data.get(image_id)
        
        if not entry or not image_path:
            continue

        article_id = entry.get("article_id", image_id)
        article_content = database.get(article_id, {}).get("content", "")

        for idx, qa_pair in enumerate(qa_list):
            # Bỏ qua nếu đã có field difficulty (Resume logic)
            if "difficulty" in qa_pair:
                continue
                
            prompt = render_difficulty_prompt(args.template_dir, article_content, qa_pair)
            
            all_qa_tasks.append({
                "image_id": image_id,
                "qa_index": idx,
                "image_path": image_path,
                "prompt": prompt
            })

    if not all_qa_tasks:
        print("[*] No QA pairs need difficulty labeling.")
        return

    print(f"[*] Total QA pairs to process: {len(all_qa_tasks)}")

    # 3. Khởi tạo Model
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 4. Batch Processing
    batch_size = args.batch_size
    pbar = tqdm(total=len(all_qa_tasks), desc="Rating Difficulty")

    for i in range(0, len(all_qa_tasks), batch_size):
        batch = all_qa_tasks[i : i + batch_size]
        batch_prompts = [t['prompt'] for t in batch]
        batch_img_paths = [t['image_path'] for t in batch]

        try:
            # Giả sử model có hàm generate_difficulty_batch (hoặc dùng chung hàm generate)
            # Nếu model trả về text, ta sẽ trim và clean
            difficulties = model.generate_difficulty_batch(
                image_paths=batch_img_paths, 
                prompts=batch_prompts
            )

            for idx, diff_label in enumerate(difficulties):
                task = batch[idx]
                img_id = task["image_id"]
                qa_idx = task["qa_index"]
                
                # Cập nhật trực tiếp vào vqa_data (dictionary đang giữ trong RAM)
                vqa_data[img_id][qa_idx]["difficulty"] = diff_label.strip()

            # Lưu định kỳ mỗi 10 batch để tránh mất dữ liệu
            if (i // batch_size) % 10 == 0:
                save_json(vqa_data, args.vqa_output_path)

        except Exception as e:
            print(f"\n[!] Error at batch {i}: {e}")
        
        pbar.update(len(batch))

    # 5. Final Save
    save_json(vqa_data, args.vqa_output_path)
    print(f"[*] Done! Difficulty added to {len(all_qa_tasks)} QA pairs.")

# Hàm main và helper (save_json, find_image_path) kế thừa từ code cũ của bạn