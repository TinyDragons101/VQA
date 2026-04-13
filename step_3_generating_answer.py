import os
import json
import argparse
import time
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel

def render_answer_prompt(template_dir, entry, article_content, questions):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    # Sử dụng template gemini để đồng bộ format prompt
    template = env.get_template("answer_generating.j2")
    
    # Giới hạn nội dung bài báo để tránh tràn token
    truncated_content = article_content[:5000] 
    
    return template.render(
        title=entry.get("title", ""),
        content=truncated_content,
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", ""),
        questions=questions
    )

def save_json(data, path):
    """Lưu file JSON an toàn bằng cách ghi vào file tạm trước"""
    temp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)

def find_image_path(image_dir, image_id):
    # Thêm .webp và các format mở rộng như file Gemini
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.webp']:
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

    # 2. Resume logic: Load output cũ (để không chạy lại những cái đã có kết quả)
    output_data = {}
    if os.path.exists(args.output_json_path):
        try:
            with open(args.output_json_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            print(f"[*] Resuming: {len(output_data)} images already answered.")
        except:
            output_data = {}

    # 3. Chuẩn bị Task
    all_tasks = []
    for image_id, questions in vqa_data.items():
        # Nếu image_id đã có trong output_data và có nội dung, bỏ qua
        if image_id in output_data and output_data[image_id]:
            continue

        entry = caption_data.get(image_id)
        image_path = find_image_path(args.image_dir, image_id)
        
        if entry and image_path:
            # Lấy list câu hỏi (chỉ lấy text nếu questions là list các dict)
            # Tùy thuộc vào cấu trúc image_questions.json của bạn
            qs_to_render = [q["question"] if isinstance(q, dict) else q for q in questions]
            
            article_id = entry.get("article_id", image_id) 
            article_info = database.get(article_id, {})
            article_content = article_info.get("content", "Không có nội dung.")
            
            prompt = render_answer_prompt(args.template_dir, entry, article_content, qs_to_render)
            
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "prompt": prompt,
                "prompt_len": len(prompt)
            })

    if not all_tasks:
        print("[*] No new images/questions to process.")
        return
    
    # Sắp xếp theo độ dài prompt để tối ưu xử lý (tương tự file Gemini)
    all_tasks.sort(key=lambda x: x['prompt_len'], reverse=args.reverse_sort)
    
    if args.limit and args.limit > 0:
        all_tasks = all_tasks[:args.limit]
        print(f"[*] Limit applied: Processing {len(all_tasks)} tasks.")

    backup_path = args.output_json_path.replace(".json", ".jsonl")
    print(f"[*] Total tasks to process: {len(all_tasks)}")

    # 4. Khởi tạo Model Local
    print(f"[*] Initializing model: {args.model_name} on {args.device}")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 5. Batch Processing
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Qwen Answering")

    try:
        # Mở file backup .jsonl ở chế độ append
        with open(backup_path, "a", encoding="utf-8") as f_backup:
            for i in range(0, len(all_tasks), batch_size):
                t0 = time.perf_counter()
                
                batch = all_tasks[i : i + batch_size]
                batch_prompts = [t['prompt'] for t in batch]
                batch_img_paths = [t['image_path'] for t in batch]
                batch_ids = [t['image_id'] for t in batch]
                
                t1 = time.perf_counter()

                try:
                    # Gọi inference của model local
                    # Lưu ý: model local thường dùng batch_img_paths và batch_prompts trực tiếp
                    batch_results = model.generate_answers_batch(
                        image_paths=batch_img_paths, 
                        prompts=batch_prompts
                    )
                    
                    t2 = time.perf_counter()

                    for idx, result in enumerate(batch_results):
                        img_id = batch_ids[idx]
                        if result:
                            # Lưu vào RAM
                            output_data[img_id] = result
                            updated_count += 1
                            
                            # Ghi ngay vào file backup .jsonl
                            line = json.dumps({img_id: result}, ensure_ascii=False)
                            f_backup.write(line + "\n")
                    
                    f_backup.flush()
                    t3 = time.perf_counter()
                    
                    print(f"""
                    ⏱️ Batch timing ({len(batch)} items):
                    - Data Prep:    {t1 - t0:.3f}s
                    - Inference:    {t2 - t1:.3f}s (Avg: {(t2-t1)/len(batch):.3f}s/img)
                    - Save/Backup:  {t3 - t2:.3f}s
                    - Total Batch:  {t3 - t0:.3f}s
                    """)

                    # Lưu file .json chính định kỳ mỗi 5 batch để an toàn
                    if (i // batch_size) % 5 == 0:
                        save_json(output_data, args.output_json_path)
                        # Dọn dẹp cache GPU cho model local
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[!] Error at batch starting with {batch_ids[0]}: {e}")
                
                pbar.update(len(batch))

    except KeyboardInterrupt:
        print("\n[!] User Interrupted (^C). Saving progress...")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
    finally:
        # Final Save
        if updated_count > 0:
            save_json(output_data, args.output_json_path)
            print(f"\n[*] Final save completed. Total entries: {len(output_data)}")
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Các đường dẫn mặc định giống file Gemini
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption_updated.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_questions.json")
    parser.add_argument("--output_json_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    
    # Model local config
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=40000)
    
    parser.add_argument("--reverse_sort", action="store_true", help="Sắp xếp prompt_len giảm dần (mặc định là tăng dần)")
    
    args = parser.parse_args()
    process_answering_pipeline(args)